from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from transformers import SiglipVisionModel

from lerobot.policies.pi05.modeling_pi05 import PI05Pytorch, make_att_2d_masks
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.utils.constants import ACTION, OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS

from .configuration_cloudedge_pi05_v3 import CloudEdgePI05V3Config
from .modeling_cloudedge_pi05 import CloudContext, CloudEdgePI05Policy


@dataclass
class CloudPlan:
    """Network-transferable result of cloud-side pi0.5 flow sampling."""

    actions: Tensor
    action_features: Tensor


class EdgeActionCorrectionHead(nn.Module):
    """Lightweight post-flow action correction that can run without the action expert."""

    def __init__(self, edge_width: int, expert_width: int, action_dim: int):
        super().__init__()
        self.edge_norm = nn.LayerNorm(edge_width)
        self.cloud_norm = nn.LayerNorm(expert_width)
        self.action_norm = nn.LayerNorm(action_dim)
        self.edge_projection = nn.Linear(edge_width, expert_width)
        self.action_projection = nn.Linear(action_dim, expert_width)
        self.fusion = nn.Sequential(
            nn.Linear(3 * expert_width, expert_width),
            nn.SiLU(),
            nn.Linear(expert_width, action_dim),
        )
        nn.init.zeros_(self.fusion[-1].weight)
        nn.init.zeros_(self.fusion[-1].bias)

    def forward(
        self, cloud_features: Tensor, edge_features: Tensor, base_actions: Tensor
    ) -> Tensor:
        cloud = self.cloud_norm(cloud_features.to(dtype=torch.float32))
        edge = self.edge_norm(edge_features.to(dtype=torch.float32))
        edge = F.silu(self.edge_projection(edge))
        edge = edge[:, None, :].expand(-1, cloud.shape[1], -1)
        action = self.action_norm(base_actions.to(dtype=torch.float32))
        action = F.silu(self.action_projection(action))
        return self.fusion(torch.cat([cloud, edge, action], dim=-1))


class CloudEdgePI05V3Pytorch(PI05Pytorch):
    """Cloud pi0.5 sampler plus a separable current-image edge correction."""

    config: CloudEdgePI05V3Config

    def __init__(self, config: CloudEdgePI05V3Config, rtc_processor=None):
        super().__init__(config, rtc_processor=rtc_processor)
        self.edge_vision = SiglipVisionModel.from_pretrained(
            config.edge_model_name, revision=config.edge_model_revision
        )
        self.edge_vision.requires_grad_(False)
        self.edge_vision.eval()
        expert_width = self.action_in_proj.out_features
        self.edge_action_head = EdgeActionCorrectionHead(
            edge_width=self.edge_vision.config.hidden_size,
            expert_width=expert_width,
            action_dim=config.max_action_dim,
        )

    def train(self, mode: bool = True):
        super().train(mode)
        self.edge_vision.eval()
        return self

    def encode_edge_images(self, images: list[Tensor], img_masks: list[Tensor]) -> Tensor:
        if not self.config.use_edge_vision:
            return torch.zeros(
                images[0].shape[0],
                self.edge_vision.config.hidden_size,
                dtype=torch.float32,
                device=images[0].device,
            )
        pooled_views = []
        with torch.no_grad():
            for image in images:
                output = self.edge_vision(pixel_values=image.to(dtype=torch.float32))
                pooled_views.append(output.pooler_output)
        pooled = torch.stack(pooled_views, dim=1)
        valid = torch.stack(img_masks, dim=1).to(dtype=pooled.dtype)
        denom = valid.sum(dim=1, keepdim=True).clamp_min(1.0)
        return (pooled * valid.unsqueeze(-1)).sum(dim=1) / denom

    def flow_training_outputs(
        self,
        images,
        img_masks,
        tokens,
        masks,
        actions,
        *,
        noise: Tensor,
        time: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        time_expanded = time[:, None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        target_velocity = noise - actions
        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            images, img_masks, tokens, masks
        )
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(
            x_t, time
        )
        if (
            self.paligemma_with_expert.paligemma.language_model.layers[
                0
            ].self_attn.q_proj.weight.dtype
            == torch.bfloat16
        ):
            prefix_embs = prefix_embs.to(dtype=torch.bfloat16)
            suffix_embs = suffix_embs.to(dtype=torch.bfloat16)

        pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
        att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)
        attention = self._prepare_attention_masks_4d(
            make_att_2d_masks(pad_masks, att_masks)
        )
        position_ids = torch.cumsum(pad_masks, dim=1) - 1

        def forward_func(prefix, suffix, attention_mask, positions, condition):
            (_, suffix_out), _ = self.paligemma_with_expert.forward(
                attention_mask=attention_mask,
                position_ids=positions,
                past_key_values=None,
                inputs_embeds=[prefix, suffix],
                use_cache=False,
                adarms_cond=[None, condition],
            )
            return suffix_out

        suffix_out = self._apply_checkpoint(
            forward_func,
            prefix_embs,
            suffix_embs,
            attention,
            position_ids,
            adarms_cond,
        )
        features = suffix_out[:, -self.config.chunk_size :].to(dtype=torch.float32)
        velocity = self.action_out_proj(features)
        flow_loss = F.mse_loss(target_velocity, velocity, reduction="none")
        action_estimate = x_t - time_expanded * velocity
        return flow_loss, features, action_estimate

    def encode_cloud_context(self, images, img_masks, tokens, masks) -> CloudContext:
        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            images, img_masks, tokens, masks
        )
        attention = self._prepare_attention_masks_4d(
            make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        )
        position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
        self.paligemma_with_expert.paligemma.language_model.config._attn_implementation = (
            "eager"
        )
        _, past_key_values = self.paligemma_with_expert.forward(
            attention_mask=attention,
            position_ids=position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=True,
        )
        return CloudContext(prefix_pad_masks, past_key_values)

    def denoise_step_with_features(
        self, cloud_context: CloudContext, x_t: Tensor, timestep: Tensor
    ) -> tuple[Tensor, Tensor]:
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(
            x_t, timestep
        )
        prefix_pad_masks = cloud_context.prefix_pad_masks
        suffix_len = suffix_pad_masks.shape[1]
        batch_size = prefix_pad_masks.shape[0]
        prefix_len = prefix_pad_masks.shape[1]
        prefix_attention = prefix_pad_masks[:, None, :].expand(
            batch_size, suffix_len, prefix_len
        )
        suffix_attention = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)
        attention = self._prepare_attention_masks_4d(
            torch.cat([prefix_attention, suffix_attention], dim=2)
        )
        position_ids = (
            torch.sum(prefix_pad_masks, dim=-1)[:, None]
            + torch.cumsum(suffix_pad_masks, dim=1)
            - 1
        )
        self.paligemma_with_expert.gemma_expert.model.config._attn_implementation = (
            "eager"
        )
        outputs, _ = self.paligemma_with_expert.forward(
            attention_mask=attention,
            position_ids=position_ids,
            past_key_values=cloud_context.past_key_values,
            inputs_embeds=[None, suffix_embs],
            use_cache=False,
            adarms_cond=[None, adarms_cond],
        )
        features = outputs[1][:, -self.config.chunk_size :].to(dtype=torch.float32)
        return self.action_out_proj(features), features

    @torch.no_grad()
    def sample_cloud_plan(
        self,
        cloud_context: CloudContext,
        noise: Tensor | None = None,
        num_steps: int | None = None,
    ) -> CloudPlan:
        num_steps = num_steps or self.config.num_inference_steps
        batch_size = cloud_context.prefix_pad_masks.shape[0]
        device = cloud_context.prefix_pad_masks.device
        if noise is None:
            noise = self.sample_noise(
                (batch_size, self.config.chunk_size, self.config.max_action_dim),
                device,
            )
        dt = -1.0 / num_steps
        x_t = noise
        final_features = None
        for step in range(num_steps):
            timestep = torch.full(
                (batch_size,),
                1.0 + step * dt,
                dtype=torch.float32,
                device=device,
            )
            velocity, final_features = self.denoise_step_with_features(
                cloud_context, x_t, timestep
            )
            x_t = x_t + dt * velocity
        assert final_features is not None
        return CloudPlan(actions=x_t, action_features=final_features)

    def correct_plan(self, plan: CloudPlan, edge_context: Tensor) -> Tensor:
        delta = self.edge_action_head(
            plan.action_features, edge_context, plan.actions
        )
        return plan.actions + delta


class CloudEdgePI05V3Policy(CloudEdgePI05Policy):
    config_class = CloudEdgePI05V3Config
    name = "cloudedge_pi05_v3"

    def __init__(self, config: CloudEdgePI05V3Config, **kwargs):
        PreTrainedPolicy.__init__(self, config)
        config.validate_features()
        self.config = config
        self.init_rtc_processor()
        self.model = CloudEdgePI05V3Pytorch(config, rtc_processor=self.rtc_processor)
        if config.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
        self.model.to(config.device)
        self.reset()

    def _get_default_peft_targets(self) -> dict[str, Any]:
        target_modules = (
            r"model\.paligemma_with_expert\."
            r"(paligemma\.model\.language_model|gemma_expert\.model)\."
            r"layers\.\d+\."
            r"(self_attn\.(q|k|v|o)_proj|mlp\.(gate|up|down)_proj)"
        )
        return {
            "target_modules": target_modules,
            "modules_to_save": ["edge_action_head"],
            "r": 8,
            "lora_alpha": 8,
            "lora_dropout": 0.0,
            "bias": "none",
        }

    def forward(self, batch: dict[str, Tensor], reduction: str = "mean"):
        current_images, current_masks, stale_images, stale_masks, delay = (
            self._training_image_views(batch)
        )
        tokens = batch[OBS_LANGUAGE_TOKENS]
        masks = batch[OBS_LANGUAGE_ATTENTION_MASK]
        actions = self.prepare_action(batch)
        edge_context = self.model.encode_edge_images(current_images, current_masks)
        noise = self.model.sample_noise(actions.shape, actions.device)
        time = self.model.sample_time(actions.shape[0], actions.device)

        fresh_flow, fresh_features, fresh_base = self.model.flow_training_outputs(
            current_images,
            current_masks,
            tokens,
            masks,
            actions,
            noise=noise,
            time=time,
        )
        stale_flow, stale_features, stale_base = self.model.flow_training_outputs(
            stale_images,
            stale_masks,
            tokens,
            masks,
            actions,
            noise=noise,
            time=time,
        )
        fresh_prediction = self.model.correct_plan(
            CloudPlan(fresh_base, fresh_features), edge_context
        )
        stale_prediction = self.model.correct_plan(
            CloudPlan(stale_base, stale_features), edge_context
        )

        action_dim = self.config.output_features[ACTION].shape[0]
        target = actions[:, :, :action_dim]
        fresh_flow = fresh_flow[:, :, :action_dim]
        stale_flow = stale_flow[:, :, :action_dim]
        fresh_correction = F.l1_loss(
            fresh_prediction[:, :, :action_dim], target, reduction="none"
        )
        stale_correction = F.l1_loss(
            stale_prediction[:, :, :action_dim], target, reduction="none"
        )
        fresh = (
            self.config.flow_loss_weight * fresh_flow
            + self.config.correction_loss_weight * fresh_correction
        )
        stale = (
            self.config.flow_loss_weight * stale_flow
            + self.config.correction_loss_weight * stale_correction
        )
        stale_weight = self._stale_weight()
        losses = (1.0 - stale_weight) * fresh + stale_weight * stale
        self.config.cloudedge_train_step += 1

        per_sample = losses.mean(dim=(1, 2))
        output = {
            "loss": per_sample.mean().item(),
            "loss_fresh": fresh.mean().item(),
            "loss_stale": stale.mean().item(),
            "loss_flow_fresh": fresh_flow.mean().item(),
            "loss_flow_stale": stale_flow.mean().item(),
            "loss_correction_fresh": fresh_correction.mean().item(),
            "loss_correction_stale": stale_correction.mean().item(),
            "stale_loss_weight": stale_weight,
            "sampled_delay_mean": delay.float().mean().item(),
        }
        if reduction == "none":
            return per_sample, output
        return per_sample.mean(), output

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        """Recompute the lightweight correction on every environment step."""
        assert not self._rtc_enabled()
        self.eval()
        return self.predict_action_chunk(batch)[:, 0]

    @torch.no_grad()
    def predict_action_chunk(
        self, batch: dict[str, Tensor], *, record_observation: bool = True, **kwargs
    ) -> Tensor:
        self.eval()
        current_images, current_masks, stale_images, stale_masks, _ = (
            self._inference_image_views(batch, record_observation=record_observation)
        )
        tokens = batch[OBS_LANGUAGE_TOKENS]
        masks = batch[OBS_LANGUAGE_ATTENTION_MASK]
        cloud_context = self.model.encode_cloud_context(
            stale_images, stale_masks, tokens, masks
        )
        plan = self.model.sample_cloud_plan(
            cloud_context,
            noise=kwargs.get("noise"),
            num_steps=kwargs.get("num_steps"),
        )
        edge_context = self.model.encode_edge_images(current_images, current_masks)
        actions = self.model.correct_plan(plan, edge_context)
        action_dim = self.config.output_features[ACTION].shape[0]
        return actions[:, :, :action_dim]

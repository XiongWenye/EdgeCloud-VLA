from collections import deque
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from transformers import SiglipVisionModel

from lerobot.policies.pi05.modeling_pi05 import (
    PI05Policy,
    PI05Pytorch,
    make_att_2d_masks,
)
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.utils.constants import (
    ACTION,
    OBS_LANGUAGE_ATTENTION_MASK,
    OBS_LANGUAGE_TOKENS,
)

from .configuration_cloudedge_pi05 import CloudEdgePI05Config


@dataclass
class CloudContext:
    """Network-transferable logical cloud result used by the edge denoiser."""

    prefix_pad_masks: Tensor
    past_key_values: Any


class EdgeResidualActionHead(nn.Module):
    """Paper-style residual action head adapted to pi0.5 flow velocity."""

    def __init__(self, edge_width: int, expert_width: int, action_dim: int):
        super().__init__()
        self.edge_norm = nn.LayerNorm(edge_width)
        self.cloud_norm = nn.LayerNorm(expert_width)
        self.edge_projection = nn.Linear(edge_width, expert_width)
        self.fusion = nn.Sequential(
            nn.Linear(2 * expert_width, expert_width),
            nn.SiLU(),
            nn.Linear(expert_width, action_dim),
        )

        # Preserve the aligned pi0.5 flow field exactly at initialization while
        # keeping the upstream edge path non-zero and trainable.
        nn.init.zeros_(self.fusion[-1].weight)
        nn.init.zeros_(self.fusion[-1].bias)

    def forward(self, cloud_features: Tensor, edge_features: Tensor) -> Tensor:
        cloud = self.cloud_norm(cloud_features.to(dtype=torch.float32))
        edge = self.edge_norm(edge_features.to(dtype=torch.float32))
        edge = F.silu(self.edge_projection(edge))
        edge = edge[:, None, :].expand(-1, cloud.shape[1], -1)
        return self.fusion(torch.cat([cloud, edge], dim=-1))


class CloudEdgePI05Pytorch(PI05Pytorch):
    """pi0.5 flow model with a frozen current-image edge condition."""

    config: CloudEdgePI05Config

    def __init__(self, config: CloudEdgePI05Config, rtc_processor=None):
        super().__init__(config, rtc_processor=rtc_processor)
        self.edge_vision = SiglipVisionModel.from_pretrained(
            config.edge_model_name, revision=config.edge_model_revision
        )
        self.edge_vision.requires_grad_(False)
        self.edge_vision.eval()

        expert_width = self.action_in_proj.out_features
        edge_width = self.edge_vision.config.hidden_size
        self.edge_action_head = EdgeResidualActionHead(
            edge_width=edge_width,
            expert_width=expert_width,
            action_dim=config.max_action_dim,
        )

    def train(self, mode: bool = True):
        super().train(mode)
        self.edge_vision.eval()
        return self

    def encode_edge_images(self, images: list[Tensor], img_masks: list[Tensor]) -> Tensor | None:
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
        pooled = (pooled * valid.unsqueeze(-1)).sum(dim=1) / denom
        return pooled

    def _predict_velocity(self, cloud_features: Tensor, edge_context: Tensor | None) -> Tensor:
        velocity = self.action_out_proj(cloud_features.to(dtype=torch.float32))
        if edge_context is not None:
            velocity = velocity + self.edge_action_head(cloud_features, edge_context)
        return velocity

    def forward(
        self,
        images,
        img_masks,
        tokens,
        masks,
        actions,
        edge_context: Tensor | None = None,
        noise=None,
        time=None,
    ) -> Tensor:
        if noise is None:
            noise = self.sample_noise(actions.shape, actions.device)
        if time is None:
            time = self.sample_time(actions.shape[0], actions.device)

        time_expanded = time[:, None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(images, img_masks, tokens, masks)
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = super().embed_suffix(x_t, time)
        if (
            self.paligemma_with_expert.paligemma.language_model.layers[0].self_attn.q_proj.weight.dtype
            == torch.bfloat16
        ):
            prefix_embs = prefix_embs.to(dtype=torch.bfloat16)
            suffix_embs = suffix_embs.to(dtype=torch.bfloat16)

        pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
        att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)
        attention = self._prepare_attention_masks_4d(make_att_2d_masks(pad_masks, att_masks))
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
            forward_func, prefix_embs, suffix_embs, attention, position_ids, adarms_cond
        )
        suffix_out = suffix_out[:, -self.config.chunk_size :].to(dtype=torch.float32)
        velocity = self._predict_velocity(suffix_out, edge_context)
        return F.mse_loss(u_t, velocity, reduction="none")

    def encode_cloud_context(self, images, img_masks, tokens, masks) -> CloudContext:
        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(images, img_masks, tokens, masks)
        attention = self._prepare_attention_masks_4d(
            make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        )
        position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
        self.paligemma_with_expert.paligemma.language_model.config._attn_implementation = "eager"
        _, past_key_values = self.paligemma_with_expert.forward(
            attention_mask=attention,
            position_ids=position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=True,
        )
        return CloudContext(prefix_pad_masks=prefix_pad_masks, past_key_values=past_key_values)

    def denoise_step_from_context(
        self,
        cloud_context: CloudContext,
        x_t: Tensor,
        timestep: Tensor,
        edge_context: Tensor | None,
    ) -> Tensor:
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = super().embed_suffix(x_t, timestep)
        prefix_pad_masks = cloud_context.prefix_pad_masks
        suffix_len = suffix_pad_masks.shape[1]
        batch_size = prefix_pad_masks.shape[0]
        prefix_len = prefix_pad_masks.shape[1]
        prefix_attention = prefix_pad_masks[:, None, :].expand(batch_size, suffix_len, prefix_len)
        suffix_attention = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)
        attention = self._prepare_attention_masks_4d(
            torch.cat([prefix_attention, suffix_attention], dim=2)
        )
        position_ids = (
            torch.sum(prefix_pad_masks, dim=-1)[:, None]
            + torch.cumsum(suffix_pad_masks, dim=1)
            - 1
        )
        self.paligemma_with_expert.gemma_expert.model.config._attn_implementation = "eager"
        outputs, _ = self.paligemma_with_expert.forward(
            attention_mask=attention,
            position_ids=position_ids,
            past_key_values=cloud_context.past_key_values,
            inputs_embeds=[None, suffix_embs],
            use_cache=False,
            adarms_cond=[None, adarms_cond],
        )
        cloud_features = outputs[1][:, -self.config.chunk_size :].to(dtype=torch.float32)
        return self._predict_velocity(cloud_features, edge_context)

    @torch.no_grad()
    def sample_actions_from_context(
        self,
        cloud_context: CloudContext,
        edge_context: Tensor | None,
        noise: Tensor | None = None,
        num_steps: int | None = None,
    ) -> Tensor:
        if num_steps is None:
            num_steps = self.config.num_inference_steps
        batch_size = cloud_context.prefix_pad_masks.shape[0]
        device = cloud_context.prefix_pad_masks.device
        if noise is None:
            noise = self.sample_noise(
                (batch_size, self.config.chunk_size, self.config.max_action_dim), device
            )
        dt = -1.0 / num_steps
        x_t = noise
        for step in range(num_steps):
            timestep = torch.full(
                (batch_size,), 1.0 + step * dt, dtype=torch.float32, device=device
            )
            velocity = self.denoise_step_from_context(
                cloud_context, x_t, timestep, edge_context
            )
            x_t = x_t + dt * velocity
        return x_t


class CloudEdgePI05Policy(PI05Policy):
    config_class = CloudEdgePI05Config
    name = "cloudedge_pi05"

    def __init__(self, config: CloudEdgePI05Config, **kwargs):
        PreTrainedPolicy.__init__(self, config)
        config.validate_features()
        self.config = config
        self.init_rtc_processor()
        self.model = CloudEdgePI05Pytorch(config, rtc_processor=self.rtc_processor)
        if config.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
        self.model.to(config.device)
        self.reset()

    def _get_default_peft_targets(self) -> dict[str, Any]:
        """LoRA cloud/planning layers; fully train only pi0.5 projections and edge head."""
        target_modules = (
            r"model\.paligemma_with_expert\."
            r"(paligemma\.model\.language_model|gemma_expert\.model)\."
            r"layers\.\d+\."
            r"(self_attn\.(q|k|v|o)_proj|mlp\.(gate|up|down)_proj)"
        )
        return {
            "target_modules": target_modules,
            "modules_to_save": [
                "action_in_proj",
                "action_out_proj",
                "time_mlp_in",
                "time_mlp_out",
                "edge_action_head",
            ],
            "r": 16,
            "lora_alpha": 16,
            "lora_dropout": 0.0,
            "bias": "none",
        }

    def reset(self):
        super().reset()
        capacity = max(self.config.history_window, self.config.eval_delay_max + 1)
        self._observation_history: deque[dict[str, Tensor]] = deque(maxlen=capacity)

    def _sample_training_delay(self, batch_size: int, history: int, device) -> Tensor:
        return torch.randint(1, history, (batch_size,), device=device)

    def _training_image_views(self, batch: dict[str, Tensor]):
        first_key = next(key for key in self.config.image_features if key in batch)
        first = batch[first_key]
        if first.ndim != 5:
            raise ValueError(
                "CloudEdge training expects [B,T,C,H,W] image histories; "
                f"received {tuple(first.shape)} for {first_key}"
            )
        batch_size, history = first.shape[:2]
        if history != self.config.history_window:
            raise ValueError(
                f"Expected history_window={self.config.history_window}, received T={history}"
            )
        delay = self._sample_training_delay(batch_size, history, first.device)
        row = torch.arange(batch_size, device=first.device)
        stale_index = history - 1 - delay
        current_batch = dict(batch)
        stale_batch = dict(batch)
        for key in self.config.image_features:
            if key in batch:
                current_batch[key] = batch[key][:, -1]
                stale_batch[key] = batch[key][row, stale_index]
        current_images, current_masks = super()._preprocess_images(current_batch)
        stale_images, stale_masks = super()._preprocess_images(stale_batch)
        return current_images, current_masks, stale_images, stale_masks, delay

    def _record_inference_observation(self, batch: dict[str, Tensor]) -> None:
        current_raw = {
            key: batch[key].detach().clone()
            for key in self.config.image_features
            if key in batch
        }
        self._observation_history.append(current_raw)

    def _inference_image_views(
        self, batch: dict[str, Tensor], *, record_observation: bool = True
    ):
        if record_observation:
            self._record_inference_observation(batch)
        if self.config.eval_delay_max == 0:
            delay = 0
        else:
            delay = int(torch.randint(1, self.config.eval_delay_max + 1, ()).item())
        history_index = max(0, len(self._observation_history) - 1 - delay)
        stale_raw = self._observation_history[history_index]
        stale_batch = dict(batch)
        stale_batch.update(stale_raw)
        current_images, current_masks = super()._preprocess_images(batch)
        stale_images, stale_masks = super()._preprocess_images(stale_batch)
        return current_images, current_masks, stale_images, stale_masks, delay

    def _stale_weight(self) -> float:
        maximum = self.config.stale_loss_weight_max
        warmup = self.config.stale_loss_warmup_steps
        if warmup == 0:
            return maximum
        return maximum * min(1.0, self.config.cloudedge_train_step / warmup)

    def forward(self, batch: dict[str, Tensor], reduction: str = "mean"):
        current_images, current_masks, stale_images, stale_masks, delay = self._training_image_views(batch)
        tokens = batch[OBS_LANGUAGE_TOKENS]
        masks = batch[OBS_LANGUAGE_ATTENTION_MASK]
        actions = self.prepare_action(batch)
        edge_context = self.model.encode_edge_images(current_images, current_masks)

        # Identical stochastic flow target for both paths isolates cloud-image age.
        noise = self.model.sample_noise(actions.shape, actions.device)
        time = self.model.sample_time(actions.shape[0], actions.device)
        fresh = self.model.forward(
            current_images,
            current_masks,
            tokens,
            masks,
            actions,
            edge_context=edge_context,
            noise=noise,
            time=time,
        )
        stale = self.model.forward(
            stale_images,
            stale_masks,
            tokens,
            masks,
            actions,
            edge_context=edge_context,
            noise=noise,
            time=time,
        )
        action_dim = self.config.output_features[ACTION].shape[0]
        fresh = fresh[:, :, :action_dim]
        stale = stale[:, :, :action_dim]
        stale_weight = self._stale_weight()
        losses = (1.0 - stale_weight) * fresh + stale_weight * stale
        self.config.cloudedge_train_step += 1

        per_sample = losses.mean(dim=(1, 2))
        output = {
            "loss": per_sample.mean().item(),
            "loss_fresh": fresh.mean().item(),
            "loss_stale": stale.mean().item(),
            "stale_loss_weight": stale_weight,
            "sampled_delay_mean": delay.float().mean().item(),
            "loss_per_dim": losses.mean(dim=(0, 1)).detach().cpu().tolist(),
        }
        if reduction == "none":
            return per_sample, output
        return per_sample.mean(), output

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        """Execute an action chunk while retaining one observation per environment step."""
        assert not self._rtc_enabled(), (
            "RTC is not supported for select_action, use it with predict_action_chunk"
        )
        self.eval()

        # The evaluator calls select_action at every environment step. Record every
        # observation even while executing a queued action chunk so eval_delay_max
        # remains measured in environment steps, not action chunks.
        self._record_inference_observation(batch)
        if len(self._action_queue) == 0:
            actions = self.predict_action_chunk(batch, record_observation=False)[
                :, : self.config.n_action_steps
            ]
            self._action_queue.extend(actions.transpose(0, 1))
        return self._action_queue.popleft()

    @torch.no_grad()
    def predict_action_chunk(
        self, batch: dict[str, Tensor], *, record_observation: bool = True, **kwargs
    ) -> Tensor:
        self.eval()
        current_images, current_masks, stale_images, stale_masks, _ = self._inference_image_views(
            batch, record_observation=record_observation
        )
        tokens = batch[OBS_LANGUAGE_TOKENS]
        masks = batch[OBS_LANGUAGE_ATTENTION_MASK]
        edge_context = self.model.encode_edge_images(current_images, current_masks)
        cloud_context = self.model.encode_cloud_context(stale_images, stale_masks, tokens, masks)
        actions = self.model.sample_actions_from_context(
            cloud_context,
            edge_context,
            noise=kwargs.get("noise"),
            num_steps=kwargs.get("num_steps"),
        )
        action_dim = self.config.output_features[ACTION].shape[0]
        return actions[:, :, :action_dim]

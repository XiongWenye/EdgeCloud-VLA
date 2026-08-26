#!/usr/bin/env python3
"""H200 gates for CloudEdge pi0.5 v2 base parity, PEFT, gradients, and reload."""

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from peft import PeftModel

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.pi05.modeling_pi05 import PI05Policy
from lerobot.utils.constants import (
    OBS_LANGUAGE_ATTENTION_MASK,
    OBS_LANGUAGE_TOKENS,
)
from lerobot_policy_cloudedge_pi05.modeling_cloudedge_pi05 import CloudEdgePI05Policy


def grad_norm(named_parameters, predicate) -> float:
    total = 0.0
    for name, parameter in named_parameters:
        if predicate(name) and parameter.grad is not None:
            total += parameter.grad.detach().float().square().sum().item()
    return total**0.5


def make_processed_batch(policy, bootstrap: Path, device: torch.device):
    preprocessor, _ = make_pre_post_processors(
        policy_cfg=policy.config,
        pretrained_path=bootstrap,
        preprocessor_overrides={"device_processor": {"device": str(device)}},
    )
    raw = {
        "observation.images.image": torch.rand(1, 3, 256, 256, device=device),
        "observation.images.image2": torch.rand(1, 3, 256, 256, device=device),
        "observation.state": torch.zeros(1, 8, device=device),
        "task": "pick up the black bowl between the plate and the ramekin and place it on the plate",
    }
    return preprocessor(raw)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bootstrap", type=Path, required=True)
    parser.add_argument("--aligned-base", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite existing smoke output: {args.output}")
    args.output.mkdir(parents=True)

    assert torch.cuda.is_available()
    device = torch.device("cuda")
    torch.manual_seed(7)

    base = PI05Policy.from_pretrained(args.aligned_base)
    base.to(device).eval()
    cloudedge = CloudEdgePI05Policy.from_pretrained(args.bootstrap)
    cloudedge.to(device).eval()

    assert cloudedge.config.architecture_version == 2
    assert cloudedge.config.freeze_vision_encoder
    assert cloudedge.config.chunk_size == 10
    assert cloudedge.config.n_action_steps == 5
    assert cloudedge.config.num_inference_steps == 10

    processed = make_processed_batch(cloudedge, args.bootstrap, device)
    tokens = processed[OBS_LANGUAGE_TOKENS]
    token_masks = processed[OBS_LANGUAGE_ATTENTION_MASK]
    base_images, base_masks = base._preprocess_images(processed)
    ce_images, ce_masks = cloudedge._preprocess_images(processed)
    noise = torch.randn(1, 10, 32, device=device)

    with torch.no_grad():
        base_chunk = base.model.sample_actions(
            base_images,
            base_masks,
            tokens,
            token_masks,
            noise=noise.clone(),
        )
        edge_context = cloudedge.model.encode_edge_images(ce_images, ce_masks)
        cloud_context = cloudedge.model.encode_cloud_context(
            ce_images,
            ce_masks,
            tokens,
            token_masks,
        )
        ce_chunk = cloudedge.model.sample_actions_from_context(
            cloud_context,
            edge_context,
            noise=noise.clone(),
        )
    parity_max_abs = (base_chunk - ce_chunk).abs().max().item()
    assert parity_max_abs <= 1e-5, parity_max_abs

    cloudedge.config.pretrained_path = args.bootstrap
    peft_policy = cloudedge.wrap_with_peft(
        peft_cli_overrides={"method_type": "LORA", "r": 16}
    )
    peft_policy.train()

    trainable = [name for name, parameter in peft_policy.named_parameters() if parameter.requires_grad]
    unexpected = [
        name
        for name in trainable
        if "lora_" not in name and "modules_to_save.default" not in name
    ]
    assert not unexpected, unexpected[:20]
    assert any("paligemma.model.language_model" in name and "lora_" in name for name in trainable)
    assert any("gemma_expert.model" in name and "lora_" in name for name in trainable)
    assert any("edge_action_head.modules_to_save.default" in name for name in trainable)
    assert not any("vision_tower" in name or "edge_vision" in name for name in trainable)

    trainable_count = sum(
        parameter.numel() for parameter in peft_policy.parameters() if parameter.requires_grad
    )
    total_count = sum(parameter.numel() for parameter in peft_policy.parameters())
    assert 0 < trainable_count < total_count

    train_batch = {
        "observation.images.image": torch.rand(1, 21, 3, 224, 224, device=device),
        "observation.images.image2": torch.rand(1, 21, 3, 224, 224, device=device),
        "observation.state": torch.zeros(1, 8, device=device),
        OBS_LANGUAGE_TOKENS: torch.zeros(1, 200, dtype=torch.long, device=device),
        OBS_LANGUAGE_ATTENTION_MASK: torch.ones(1, 200, dtype=torch.bool, device=device),
        "action": torch.rand(1, 10, 7, device=device) * 2.0 - 1.0,
    }
    optimizer = torch.optim.AdamW(
        [parameter for parameter in peft_policy.parameters() if parameter.requires_grad],
        lr=1e-4,
        betas=(0.9, 0.95),
        weight_decay=0.0,
    )

    optimizer.zero_grad(set_to_none=True)
    loss1, info1 = peft_policy.forward(train_batch)
    assert torch.isfinite(loss1)
    loss1.backward()
    final_grad_step1 = grad_norm(
        peft_policy.named_parameters(),
        lambda name: "edge_action_head" in name and "fusion.2.weight" in name,
    )
    assert final_grad_step1 > 0.0
    optimizer.step()

    optimizer.zero_grad(set_to_none=True)
    loss2, info2 = peft_policy.forward(train_batch)
    assert torch.isfinite(loss2)
    loss2.backward()
    edge_grad_step2 = grad_norm(
        peft_policy.named_parameters(),
        lambda name: "edge_action_head" in name and "edge_projection.weight" in name,
    )
    assert edge_grad_step2 > 0.0
    assert grad_norm(
        peft_policy.named_parameters(),
        lambda name: "vision_tower" in name or "edge_vision" in name,
    ) == 0.0
    optimizer.step()
    assert peft_policy.config.cloudedge_train_step == 2

    peft_policy.eval()
    peft_policy.reset()
    with torch.no_grad():
        before_reload = peft_policy.predict_action_chunk(processed, noise=noise.clone())

    adapter_dir = args.output / "adapter_step_2"
    peft_policy.save_pretrained(adapter_dir)
    # PEFT persists adapter metadata and weights only. LeRobot's checkpoint
    # writer saves the underlying policy config separately; mirror that here
    # so this smoke exercises the exact reload contract used by evaluation.
    peft_policy.config.save_pretrained(adapter_dir)
    assert (adapter_dir / "adapter_model.safetensors").is_file()
    assert (adapter_dir / "adapter_config.json").is_file()
    assert (adapter_dir / "config.json").is_file()

    saved_config = PreTrainedConfig.from_pretrained(adapter_dir)
    assert saved_config.cloudedge_train_step == 2
    reloaded_base = CloudEdgePI05Policy.from_pretrained(
        args.bootstrap,
        config=saved_config,
    )
    reloaded = PeftModel.from_pretrained(reloaded_base, adapter_dir)
    reloaded.to(device).eval()
    reloaded.reset()
    with torch.no_grad():
        after_reload = reloaded.predict_action_chunk(processed, noise=noise.clone())
    reload_max_abs = (before_reload - after_reload).abs().max().item()
    assert reload_max_abs <= 1e-5, reload_max_abs

    report = {
        "gpu": torch.cuda.get_device_name(0),
        "base_parity_max_abs": parity_max_abs,
        "trainable_parameters": trainable_count,
        "total_parameters": total_count,
        "trainable_fraction": trainable_count / total_count,
        "trainable_tensor_count": len(trainable),
        "loss_step_1": loss1.item(),
        "loss_step_2": loss2.item(),
        "loss_info_step_1": info1,
        "loss_info_step_2": info2,
        "final_projection_grad_norm_step_1": final_grad_step1,
        "edge_projection_grad_norm_step_2": edge_grad_step2,
        "reload_max_abs": reload_max_abs,
        "cloudedge_train_step": saved_config.cloudedge_train_step,
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    print("CLOUDEDGE_V2_H200_SMOKE_SUCCESS=true")


if __name__ == "__main__":
    main()

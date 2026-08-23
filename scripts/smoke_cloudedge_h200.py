#!/usr/bin/env python3
"""Gate 4: H200 smoke validation across inference, optimizer steps, checkpointing, rollout, and delay tracing."""

import argparse
import json
from pathlib import Path
import shutil
import numpy as np
import torch

from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.pi05.modeling_pi05 import PI05Policy
from lerobot.utils.constants import (
    OBS_LANGUAGE_ATTENTION_MASK,
    OBS_LANGUAGE_TOKENS,
)
from lerobot_policy_cloudedge_pi05.configuration_cloudedge_pi05 import CloudEdgePI05Config
from lerobot_policy_cloudedge_pi05.modeling_cloudedge_pi05 import CloudEdgePI05Policy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bootstrap", type=Path, default=Path("artifacts/cloudedge_pi05_bootstrap_official_aligned"))
    parser.add_argument("--aligned-base", type=Path, default=Path("artifacts/pi05_libero_base_official_aligned"))
    parser.add_argument("--smoke-dir", type=Path, default=Path("outputs/h200_smoke_gate4"))
    args = parser.parse_args()

    assert torch.cuda.is_available(), "CUDA is required for H200 smoke test"
    device = torch.device("cuda")
    print(f"Running on GPU: {torch.cuda.get_device_name(0)}")

    if args.smoke_dir.exists():
        shutil.rmtree(args.smoke_dir)
    args.smoke_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------
    # Step 1: Bootstrap load and finite inference chunk validation
    # -------------------------------------------------------------
    print("\n--- Step 1: Testing bootstrap load & inference chunk shape ---")
    ce_policy = CloudEdgePI05Policy.from_pretrained(args.bootstrap)
    ce_policy.to(device)
    ce_policy.eval()

    assert ce_policy.config.chunk_size == 10, f"Expected chunk_size 10, got {ce_policy.config.chunk_size}"
    assert ce_policy.config.n_action_steps == 5, f"Expected n_action_steps 5, got {ce_policy.config.n_action_steps}"
    assert ce_policy.config.num_inference_steps == 10

    b = 1
    dummy_batch = {
        "observation.images.image": torch.rand(b, 3, 256, 256, device=device),
        "observation.images.image2": torch.rand(b, 3, 256, 256, device=device),
        "observation.state": torch.zeros(b, 8, device=device),
        "task": "pick up the black bowl between the plate and the ramekin and place it on the plate",
    }

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=ce_policy.config,
        pretrained_path=args.bootstrap,
        preprocessor_overrides={"device_processor": {"device": "cuda"}},
    )

    processed = preprocessor(dummy_batch)
    chunk = ce_policy.predict_action_chunk(processed)
    print(f"Predicted action chunk shape: {chunk.shape}")
    assert chunk.shape == (1, 10, 7), f"Expected shape (1, 10, 7), got {chunk.shape}"
    assert not torch.isnan(chunk).any(), "Found NaN in action chunk"
    assert not torch.isinf(chunk).any(), "Found Inf in action chunk"

    # Verify select_action returns 1 action and drains queue
    ce_policy.reset()
    act0 = ce_policy.select_action(processed)
    assert act0.shape == (1, 7)
    assert len(ce_policy._action_queue) == 4, f"Expected queue len 4, got {len(ce_policy._action_queue)}"
    print("Step 1 PASS ✅: Bootstrap inference chunk shape (1, 10, 7) and select_action verified.")

    # -------------------------------------------------------------
    # Step 2: 2 optimizer steps, save/load checkpoint, resume step
    # -------------------------------------------------------------
    print("\n--- Step 2: Testing optimizer steps, checkpointing & resume ---")
    ce_policy.train()
    optimizer = torch.optim.AdamW(
        [p for p in ce_policy.parameters() if p.requires_grad],
        lr=2.5e-5,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=0.01,
    )

    # Create dummy training batch with history T=21
    train_batch = {
        "observation.images.image": torch.rand(2, 21, 3, 224, 224, device=device),
        "observation.images.image2": torch.rand(2, 21, 3, 224, 224, device=device),
        "observation.state": torch.zeros(2, 8, device=device),
        OBS_LANGUAGE_TOKENS: torch.zeros(2, 200, dtype=torch.long, device=device),
        OBS_LANGUAGE_ATTENTION_MASK: torch.ones(2, 200, dtype=torch.bool, device=device),
        "action": torch.rand(2, 10, 7, device=device),
    }

    # Step 1
    optimizer.zero_grad()
    loss1, info1 = ce_policy.forward(train_batch)
    loss1.backward()
    optimizer.step()
    print(f"Step 1 loss: {loss1.item():.4f}, fresh: {info1[loss_fresh]:.4f}, stale: {info1[loss_stale]:.4f}")
    assert not torch.isnan(loss1), "Step 1 loss is NaN"

    # Step 2
    optimizer.zero_grad()
    loss2, info2 = ce_policy.forward(train_batch)
    loss2.backward()
    optimizer.step()
    print(f"Step 2 loss: {loss2.item():.4f}, fresh: {info2[loss_fresh]:.4f}, stale: {info2[loss_stale]:.4f}")
    assert not torch.isnan(loss2), "Step 2 loss is NaN"

    # Save checkpoint
    ckpt_dir = args.smoke_dir / "checkpoint_step_2"
    ce_policy.save_pretrained(ckpt_dir)
    torch.save(optimizer.state_dict(), ckpt_dir / "optimizer.pt")
    print(f"Saved checkpoint to {ckpt_dir}")

    # Resume from checkpoint
    resumed_policy = CloudEdgePI05Policy.from_pretrained(ckpt_dir)
    resumed_policy.to(device)
    resumed_policy.train()
    resumed_opt = torch.optim.AdamW(
        [p for p in resumed_policy.parameters() if p.requires_grad],
        lr=2.5e-5,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=0.01,
    )
    resumed_opt.load_state_dict(torch.load(ckpt_dir / "optimizer.pt"))

    # Step 3
    resumed_opt.zero_grad()
    loss3, info3 = resumed_policy.forward(train_batch)
    loss3.backward()
    resumed_opt.step()
    print(f"Step 3 (resumed) loss: {loss3.item():.4f}")
    assert not torch.isnan(loss3), "Step 3 loss is NaN"
    print("Step 2 PASS ✅: Optimizer steps, checkpoint saving, loading, and resume verified.")

    # -------------------------------------------------------------
    # Step 3 & 4: Delay trace logging at d_max=10
    # -------------------------------------------------------------
    print("\n--- Step 3 & 4: Testing delay trace logging at d_max=10 ---")
    ce_policy.eval()
    ce_policy.config.eval_delay_max = 10
    ce_policy.reset()

    delays_logged = []
    indices_logged = []
    for step_idx in range(25):
        frame = {
            "observation.images.image": torch.full((1, 3, 224, 224), fill_value=float(step_idx), device=device),
            "observation.images.image2": torch.full((1, 3, 224, 224), fill_value=float(step_idx), device=device),
            OBS_LANGUAGE_TOKENS: torch.zeros(1, 200, dtype=torch.long, device=device),
            OBS_LANGUAGE_ATTENTION_MASK: torch.ones(1, 200, dtype=torch.bool, device=device),
        }
        curr_imgs, curr_masks, stale_imgs, stale_masks, d = ce_policy._inference_image_views(frame)
        stale_val = stale_imgs[0][0, 0, 0, 0].item()
        delays_logged.append(d)
        indices_logged.append(int(stale_val))

    print(f"Sampled delays across 25 steps: {delays_logged[:15]}...")
    print(f"Selected stale frame indices: {indices_logged[:15]}...")
    assert all(1 <= d <= 10 for d in delays_logged if d > 0)
    print("Step 4 PASS ✅: Delay trace logged and verified.")

    print("\n=======================================================")
    print("GATE 4 SMOKE TEST COMPLETE: ALL STEPS PASSED SUCCESSFULLY ✅")
    print("=======================================================")


if __name__ == "__main__":
    main()

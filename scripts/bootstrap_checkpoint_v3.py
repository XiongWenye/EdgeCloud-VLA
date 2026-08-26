#!/usr/bin/env python3
"""Create a zero-correction CloudEdge pi0.5 v3 bootstrap from the aligned base."""

import argparse
import dataclasses
import json
import os
from pathlib import Path

import torch
from lerobot.policies.pi05.modeling_pi05 import PI05Policy

from lerobot_policy_cloudedge_pi05.configuration_cloudedge_pi05_v3 import (
    CloudEdgePI05V3Config,
)
from lerobot_policy_cloudedge_pi05.modeling_cloudedge_pi05_v3 import (
    CloudEdgePI05V3Policy,
)
from lerobot_policy_cloudedge_pi05.processor_cloudedge_pi05_v3 import (
    make_cloudedge_pi05_v3_pre_post_processors,
)


def parse_args():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base", type=Path, default=root / "artifacts/pi05_libero_base_official_aligned"
    )
    parser.add_argument(
        "--norm-stats",
        type=Path,
        default=root
        / "cache/openpi_reference/openpi-assets/checkpoints/pi05_libero/assets/physical-intelligence/libero/norm_stats.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "artifacts/cloudedge_pi05_v3_bootstrap_official_aligned",
    )
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def load_norm_stats(path: Path):
    source = json.loads(path.read_text())["norm_stats"]
    mapping = {"state": "observation.state", "actions": "action"}
    return {
        mapping[feature]: {
            name: torch.tensor(values, dtype=torch.float32)
            for name, values in statistics.items()
        }
        for feature, statistics in source.items()
    }


def main():
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")

    base = PI05Policy.from_pretrained(args.base)
    fields = {field.name for field in dataclasses.fields(CloudEdgePI05V3Config)}
    init = {key: getattr(base.config, key) for key in fields if hasattr(base.config, key)}
    init.update(
        {
            "device": "cpu",
            "n_action_steps": 1,
            "freeze_vision_encoder": True,
            "history_window": 21,
            "stale_loss_weight_max": 0.5,
            "stale_loss_warmup_steps": 2_500,
            "flow_loss_weight": 1.0,
            "correction_loss_weight": 1.0,
            "cloudedge_train_step": 0,
            "push_to_hub": False,
        }
    )
    config = CloudEdgePI05V3Config(**init)
    config.input_features = base.config.input_features
    config.output_features = base.config.output_features
    policy = CloudEdgePI05V3Policy(config)
    missing, unexpected = policy.load_state_dict(base.state_dict(), strict=False)
    allowed = ("model.edge_vision.", "model.edge_action_head.")
    bad_missing = [key for key in missing if not key.startswith(allowed)]
    if bad_missing or unexpected:
        raise RuntimeError(f"Bad base transfer: missing={bad_missing}, unexpected={unexpected}")

    head = policy.model.edge_action_head
    assert torch.count_nonzero(head.fusion[-1].weight).item() == 0
    assert torch.count_nonzero(head.edge_projection.weight).item() > 0
    config.device = args.device
    policy.config.device = args.device
    preprocessor, postprocessor = make_cloudedge_pi05_v3_pre_post_processors(
        config, dataset_stats=load_norm_stats(args.norm_stats)
    )
    args.output.mkdir(parents=True)
    policy.save_pretrained(args.output)
    preprocessor.save_pretrained(args.output)
    postprocessor.save_pretrained(args.output)
    tokenizer = args.base / "paligemma_tokenizer.model"
    if tokenizer.exists():
        os.link(tokenizer, args.output / tokenizer.name)
    print(f"CLOUDEDGE_V3_BOOTSTRAP_SUCCESS={args.output}")
    print(f"base_shared_tensors={len(base.state_dict())}")
    print(f"newly_initialized_tensors={len(missing)}")


if __name__ == "__main__":
    main()

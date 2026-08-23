#!/usr/bin/env python3
"""Create an aligned CloudEdge checkpoint initialized from artifacts/pi05_libero_base_official_aligned."""

import argparse
import dataclasses
import json
import os
from pathlib import Path
import shutil
import torch
from lerobot.policies.pi05.modeling_pi05 import PI05Policy

from lerobot_policy_cloudedge_pi05.configuration_cloudedge_pi05 import CloudEdgePI05Config
from lerobot_policy_cloudedge_pi05.modeling_cloudedge_pi05 import CloudEdgePI05Policy
from lerobot_policy_cloudedge_pi05.processor_cloudedge_pi05 import (
    make_cloudedge_pi05_pre_post_processors,
)


def parse_args():
    ROOT = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base",
        type=Path,
        default=ROOT / "artifacts/pi05_libero_base_official_aligned",
        help="Path to aligned base artifact",
    )
    parser.add_argument(
        "--norm-stats",
        type=Path,
        default=ROOT
        / "cache/openpi_reference/openpi-assets/checkpoints/pi05_libero/assets/physical-intelligence/libero/norm_stats.json",
        help="Path to norm_stats.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "artifacts/cloudedge_pi05_bootstrap_official_aligned",
        help="Path to output bootstrap artifact",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--history-window", type=int, default=21)
    parser.add_argument("--lambda-max", type=float, default=0.5)
    parser.add_argument("--warmup-steps", type=int, default=10_000)
    return parser.parse_args()


def load_openpi_norm_stats(path: Path) -> dict[str, dict[str, torch.Tensor]]:
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
    output = Path(args.output)
    if output.exists():
        print(f"Directory {output} exists, resetting...")
        shutil.rmtree(output)

    dataset_stats = load_openpi_norm_stats(args.norm_stats)
    base = PI05Policy.from_pretrained(args.base)
    valid_fields = {field.name for field in dataclasses.fields(CloudEdgePI05Config)}
    init = {key: getattr(base.config, key) for key in valid_fields if hasattr(base.config, key)}
    init.update(
        {
            "device": "cpu",  # initialize on cpu for memory efficiency during assembly
            "history_window": args.history_window,
            "stale_loss_weight_max": args.lambda_max,
            "stale_loss_warmup_steps": args.warmup_steps,
            "push_to_hub": False,
        }
    )
    config = CloudEdgePI05Config(**init)
    config.input_features = base.config.input_features
    config.output_features = base.config.output_features

    assert config.chunk_size == 10, f"Expected chunk_size 10, got {config.chunk_size}"
    assert config.n_action_steps == 5, f"Expected n_action_steps 5, got {config.n_action_steps}"
    assert config.num_inference_steps == 10, f"Expected num_inference_steps 10, got {config.num_inference_steps}"

    policy = CloudEdgePI05Policy(config)
    missing, unexpected = policy.load_state_dict(base.state_dict(), strict=False)
    expected_prefixes = (
        "model.edge_vision.",
        "model.edge_projection.",
        "model.cloudedge_train_step",
    )
    bad_missing = [key for key in missing if not key.startswith(expected_prefixes)]
    if bad_missing or unexpected:
        raise RuntimeError(f"Bad base transfer: missing={bad_missing}, unexpected={unexpected}")

    # Ensure edge projection is zero-initialized
    with torch.no_grad():
        if hasattr(policy.model, "edge_projection"):
            for p in policy.model.edge_projection.parameters():
                torch.nn.init.zeros_(p)

    # Set target device for runtime
    config.device = args.device
    policy.config.device = args.device

    preprocessor, postprocessor = make_cloudedge_pi05_pre_post_processors(
        config, dataset_stats=dataset_stats
    )
    output.mkdir(parents=True, exist_ok=True)
    policy.save_pretrained(output)
    preprocessor.save_pretrained(output)
    postprocessor.save_pretrained(output)

    # Link tokenizer model
    tok_model = args.base / "paligemma_tokenizer.model"
    if tok_model.exists():
        if (output / "paligemma_tokenizer.model").exists():
            os.remove(output / "paligemma_tokenizer.model")
        os.link(tok_model, output / "paligemma_tokenizer.model")
        print("Linked paligemma_tokenizer.model into bootstrap directory")

    print(f"BOOTSTRAP_SUCCESS: checkpoint={output}")
    print(f"base_shared_tensors={len(base.state_dict())}")
    print(f"newly_initialized_tensors={len(missing)}")


if __name__ == "__main__":
    main()

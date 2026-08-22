#!/usr/bin/env python3
"""Create a CloudEdge checkpoint initialized exactly from a pi0.5 checkpoint."""

import argparse
import dataclasses
from pathlib import Path

from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.policies.pi05.modeling_pi05 import PI05Policy

from lerobot_policy_cloudedge_pi05.configuration_cloudedge_pi05 import CloudEdgePI05Config
from lerobot_policy_cloudedge_pi05.modeling_cloudedge_pi05 import CloudEdgePI05Policy
from lerobot_policy_cloudedge_pi05.processor_cloudedge_pi05 import (
    make_cloudedge_pi05_pre_post_processors,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="lerobot/pi05_libero_base")
    parser.add_argument("--base-revision", default="a217bfd3b14673cf2ce597e69997ab21866438dd")
    parser.add_argument("--dataset", default="lerobot/libero")
    parser.add_argument("--dataset-revision", default="a1aaacb7f6cd6ee5fb43120f673cebb0cfea7dd4")
    parser.add_argument("--dataset-root", default=None)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--history-window", type=int, default=21)
    parser.add_argument("--lambda-max", type=float, default=0.5)
    parser.add_argument("--warmup-steps", type=int, default=10_000)
    return parser.parse_args()


def main():
    args = parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")

    metadata = LeRobotDatasetMetadata(
        args.dataset, root=args.dataset_root, revision=args.dataset_revision
    )
    base = PI05Policy.from_pretrained(args.base, revision=args.base_revision)
    valid_fields = {field.name for field in dataclasses.fields(CloudEdgePI05Config)}
    init = {key: getattr(base.config, key) for key in valid_fields if hasattr(base.config, key)}
    init.update(
        {
            "device": args.device,
            "history_window": args.history_window,
            "stale_loss_weight_max": args.lambda_max,
            "stale_loss_warmup_steps": args.warmup_steps,
            "push_to_hub": False,
        }
    )
    config = CloudEdgePI05Config(**init)
    config.input_features = base.config.input_features
    config.output_features = base.config.output_features
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

    preprocessor, postprocessor = make_cloudedge_pi05_pre_post_processors(
        config, dataset_stats=metadata.stats
    )
    output.mkdir(parents=True)
    policy.save_pretrained(output)
    preprocessor.save_pretrained(output)
    postprocessor.save_pretrained(output)
    print(f"bootstrap_checkpoint={output}")
    print(f"base_shared_tensors={len(base.state_dict())}")
    print(f"expected_new_tensors={len(missing)}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Assemble a lightweight evaluation checkpoint without duplicating model weights."""

import argparse
import json
import os
import shutil
from pathlib import Path


MEAN_STD_MAPPING = {
    "ACTION": "MEAN_STD",
    "STATE": "MEAN_STD",
    "VISUAL": "IDENTITY",
}

PROCESSOR_FILES = (
    "policy_preprocessor.json",
    "policy_postprocessor.json",
    "policy_preprocessor_step_2_normalizer_processor.safetensors",
    "policy_postprocessor_step_0_unnormalizer_processor.safetensors",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--processors", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    for path in (args.source / "config.json", args.source / "model.safetensors"):
        if not path.is_file():
            raise FileNotFoundError(path)
    for name in PROCESSOR_FILES:
        path = args.processors / name
        if not path.is_file():
            raise FileNotFoundError(path)

    args.output.mkdir(parents=True)
    config = json.loads((args.source / "config.json").read_text())
    config["normalization_mapping"] = MEAN_STD_MAPPING
    (args.output / "config.json").write_text(json.dumps(config, indent=2) + "\n")

    # Hub snapshots expose weights through relative symlinks. Resolve the blob first
    # so the output receives a valid hard link instead of a broken copied symlink.
    model_source = (args.source / "model.safetensors").resolve(strict=True)
    os.link(model_source, args.output / "model.safetensors")
    for name in PROCESSOR_FILES:
        shutil.copy2(args.processors / name, args.output / name)

    print(f"checkpoint={args.output}")
    print(f"model_inode={os.stat(args.output / 'model.safetensors').st_ino}")


if __name__ == "__main__":
    main()

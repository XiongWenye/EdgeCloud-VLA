#!/usr/bin/env python3
"""Attach original OpenPI LIBERO quantiles to vanilla and CloudEdge weights."""

import argparse
import json
import os
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file


QUANTILE_MAPPING = {
    "ACTION": "QUANTILES",
    "STATE": "QUANTILES",
    "VISUAL": "IDENTITY",
}


def update_stats(
    tensors: dict[str, torch.Tensor], raw_stats: dict, *, include_state: bool
) -> dict[str, torch.Tensor]:
    result = dict(tensors)
    features = {"actions": "action"}
    if include_state:
        features["state"] = "observation.state"
    for source_name, target_name in features.items():
        for statistic, values in raw_stats[source_name].items():
            result[f"{target_name}.{statistic}"] = torch.tensor(values, dtype=torch.float32)
    return result


def assemble(
    source: Path,
    processor_source: Path,
    output: Path,
    raw_stats: dict,
    *,
    cloudedge: bool,
) -> None:
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    output.mkdir(parents=True)

    config = json.loads((source / "config.json").read_text())
    config["device"] = "cuda"
    config["normalization_mapping"] = QUANTILE_MAPPING
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    os.link((source / "model.safetensors").resolve(strict=True), output / "model.safetensors")

    pre_json = json.loads((processor_source / "policy_preprocessor.json").read_text())
    if not cloudedge:
        pre_json["steps"] = [
            step
            for step in pre_json["steps"]
            if step["registry_name"] != "cloudedge_select_current_state_processor"
        ]
    normalizer = next(
        step for step in pre_json["steps"] if step["registry_name"] == "normalizer_processor"
    )
    normalizer["config"]["norm_map"] = QUANTILE_MAPPING
    pre_index = pre_json["steps"].index(normalizer)
    pre_state_name = f"policy_preprocessor_step_{pre_index}_normalizer_processor.safetensors"
    normalizer["state_file"] = pre_state_name
    (output / "policy_preprocessor.json").write_text(json.dumps(pre_json, indent=2) + "\n")

    post_json = json.loads((processor_source / "policy_postprocessor.json").read_text())
    unnormalizer = next(
        step for step in post_json["steps"] if step["registry_name"] == "unnormalizer_processor"
    )
    unnormalizer["config"]["norm_map"] = QUANTILE_MAPPING
    post_state_name = unnormalizer["state_file"]
    (output / "policy_postprocessor.json").write_text(json.dumps(post_json, indent=2) + "\n")

    pre_source = processor_source / "policy_preprocessor_step_3_normalizer_processor.safetensors"
    post_source = processor_source / post_state_name
    save_file(update_stats(load_file(pre_source), raw_stats, include_state=True), output / pre_state_name)
    save_file(
        update_stats(load_file(post_source), raw_stats, include_state=False),
        output / post_state_name,
    )
    print(f"checkpoint={output} type={config['type']} normalization={QUANTILE_MAPPING}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-source", type=Path, required=True)
    parser.add_argument("--cloudedge-source", type=Path, required=True)
    parser.add_argument("--stats", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    raw_stats = json.loads(args.stats.read_text())["norm_stats"]
    assemble(
        args.base_source,
        args.cloudedge_source,
        args.output_root / "pi05_libero_base_openpi_norm",
        raw_stats,
        cloudedge=False,
    )
    assemble(
        args.cloudedge_source,
        args.cloudedge_source,
        args.output_root / "cloudedge_pi05_bootstrap_openpi_norm",
        raw_stats,
        cloudedge=True,
    )


if __name__ == "__main__":
    main()

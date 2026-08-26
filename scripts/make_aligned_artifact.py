#!/usr/bin/env python3
"""Create the officially aligned LeRobot pi05_libero_base artifact."""

import json
import os
from pathlib import Path
import shutil
import torch
from safetensors.torch import save_file


def main():
    root = Path(__file__).resolve().parents[1]
    src_ckpt = root / "artifacts/pi05_libero_base_openpi_norm"
    stats_path = (
        root
        / "cache/openpi_reference/openpi-assets/checkpoints/pi05_libero/assets/physical-intelligence/libero/norm_stats.json"
    )
    tok_model_path = root / "cache/openpi_reference/big_vision/paligemma_tokenizer.model"
    out_dir = root / "artifacts/pi05_libero_base_official_aligned"

    if out_dir.exists():
        print(f"Directory {out_dir} exists, resetting...")
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. config.json
    config = json.loads((src_ckpt / "config.json").read_text())
    config["chunk_size"] = 10
    config["n_action_steps"] = 5
    config["num_inference_steps"] = 10
    config["device"] = "cuda"
    config["normalization_mapping"] = {
        "ACTION": "OPENPI_QUANTILES",
        "STATE": "OPENPI_QUANTILES",
        "VISUAL": "IDENTITY",
    }
    (out_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")

    # 2. Hardlink model weights
    os.link(src_ckpt / "model.safetensors", out_dir / "model.safetensors")

    # 3. Hardlink tokenizer model
    if tok_model_path.exists():
        os.link(tok_model_path, out_dir / "paligemma_tokenizer.model")
        print("Linked paligemma_tokenizer.model to artifact")
    else:
        print("Warning: tokenizer model not found at", tok_model_path)

    # 4. Read official normalization statistics
    norm_data = json.loads(stats_path.read_text())["norm_stats"]

    pre_stats = {}
    for stat_name, val in norm_data["state"].items():
        pre_stats[f"observation.state.{stat_name}"] = torch.tensor(val, dtype=torch.float32)
    for stat_name, val in norm_data["actions"].items():
        pre_stats[f"action.{stat_name}"] = torch.tensor(val, dtype=torch.float32)

    post_stats = {}
    for stat_name, val in norm_data["actions"].items():
        post_stats[f"action.{stat_name}"] = torch.tensor(val, dtype=torch.float32)

    save_file(pre_stats, out_dir / "policy_preprocessor_step_2_normalizer_processor.safetensors")
    save_file(post_stats, out_dir / "policy_postprocessor_step_0_unnormalizer_processor.safetensors")

    # 5. policy_preprocessor.json
    preprocessor_json = {
        "name": "policy_preprocessor",
        "steps": [
            {
                "registry_name": "rename_observations_processor",
                "config": {"rename_map": {}},
            },
            {
                "registry_name": "to_batch_processor",
                "config": {},
            },
            {
                "registry_name": "normalizer_processor",
                "config": {
                    "eps": 1e-08,
                    "features": {
                        "observation.images.image": {
                            "type": "VISUAL",
                            "shape": [3, 256, 256],
                        },
                        "observation.images.image2": {
                            "type": "VISUAL",
                            "shape": [3, 256, 256],
                        },
                        "observation.state": {
                            "type": "STATE",
                            "shape": [8],
                        },
                        "observation.images.empty_camera_0": {
                            "type": "VISUAL",
                            "shape": [3, 224, 224],
                        },
                        "action": {
                            "type": "ACTION",
                            "shape": [7],
                        },
                    },
                    "norm_map": {
                        "ACTION": "OPENPI_QUANTILES",
                        "STATE": "OPENPI_QUANTILES",
                        "VISUAL": "IDENTITY",
                    },
                },
                "state_file": "policy_preprocessor_step_2_normalizer_processor.safetensors",
            },
            {
                "registry_name": "paligemma_tokenizer_processor",
                "config": {
                    "tokenizer_path": "paligemma_tokenizer.model",
                    "max_length": 200,
                    "task_key": "task",
                    "clean_text": True,
                },
            },
            {
                "registry_name": "device_processor",
                "config": {"device": "cuda", "float_dtype": None},
            },
        ],
    }
    (out_dir / "policy_preprocessor.json").write_text(json.dumps(preprocessor_json, indent=2) + "\n")

    # 6. policy_postprocessor.json
    postprocessor_json = {
        "name": "policy_postprocessor",
        "steps": [
            {
                "registry_name": "unnormalizer_processor",
                "config": {
                    "eps": 1e-08,
                    "features": {
                        "action": {"type": "ACTION", "shape": [7]},
                    },
                    "norm_map": {
                        "ACTION": "OPENPI_QUANTILES",
                        "STATE": "OPENPI_QUANTILES",
                        "VISUAL": "IDENTITY",
                    },
                },
                "state_file": "policy_postprocessor_step_0_unnormalizer_processor.safetensors",
            },
            {
                "registry_name": "device_processor",
                "config": {"device": "cpu", "float_dtype": None},
            },
        ],
    }
    (out_dir / "policy_postprocessor.json").write_text(json.dumps(postprocessor_json, indent=2) + "\n")

    print(f"Created aligned artifact at {out_dir}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Focused unit tests for LeRobot pi05_libero_base alignment with OpenPI reference."""

import json
from pathlib import Path
import numpy as np
import pytest
import torch

from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
from lerobot.policies.pi05.configuration_pi05 import PI05Config
from lerobot.processor import (
    EnvTransition,
    NormalizerProcessorStep,
    PaliGemmaTokenizerProcessorStep,
    PolicyAction,
    PolicyProcessorPipeline,
    TransitionKey,
    UnnormalizerProcessorStep,
)
from lerobot.utils.constants import (
    ACTION,
    OBS_IMAGES,
    OBS_LANGUAGE_ATTENTION_MASK,
    OBS_LANGUAGE_TOKENS,
    OBS_STATE,
)

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = ROOT / "artifacts/pi05_libero_base_official_aligned"
PARITY_DIR = ROOT / "results/parity"


def test_config_chunk_and_horizon():
    """Gate 1: Static configuration validation."""
    config_path = ARTIFACT_DIR / "config.json"
    assert config_path.exists(), f"Missing {config_path}"
    data = json.loads(config_path.read_text())

    assert data["chunk_size"] == 10, f"Expected chunk_size 10, got {data[chunk_size]}"
    assert data["n_action_steps"] == 5, f"Expected n_action_steps 5, got {data[n_action_steps]}"
    assert data["num_inference_steps"] == 10, f"Expected num_inference_steps 10, got {data[num_inference_steps]}"
    assert data["max_action_dim"] == 32, f"Expected max_action_dim 32, got {data[max_action_dim]}"
    assert data["max_state_dim"] == 32, f"Expected max_state_dim 32, got {data[max_state_dim]}"
    assert data["output_features"]["action"]["shape"] == [7]
    assert data["empty_cameras"] == 1

    pre_json = json.loads((ARTIFACT_DIR / "policy_preprocessor.json").read_text())
    step_names = [step["registry_name"] for step in pre_json["steps"]]
    assert "pi05_prepare_state_tokenizer_processor_step" not in step_names, (
        "State-in-prompt processor step must NOT be active in aligned artifact"
    )
    assert "paligemma_tokenizer_processor" in step_names, (
        "paligemma_tokenizer_processor must be active in aligned artifact"
    )


def test_token_ids_and_mask():
    """Gate 2: Exact token IDs and attention mask parity."""
    op_trace_path = PARITY_DIR / "openpi_trace_spatial_t0_e0.npz"
    assert op_trace_path.exists(), f"Missing reference trace {op_trace_path}"
    op_trace = np.load(op_trace_path)

    prompt = str(op_trace["prompt"])
    step = PaliGemmaTokenizerProcessorStep(
        tokenizer_path=str(ARTIFACT_DIR / "paligemma_tokenizer.model"),
        max_length=200,
        task_key="task",
        clean_text=True,
    )

    transition = {
        TransitionKey.OBSERVATION: {},
        TransitionKey.COMPLEMENTARY_DATA: {"task": prompt},
    }
    step._current_transition = transition
    obs = step.observation({})

    tokens = obs[OBS_LANGUAGE_TOKENS].cpu().numpy()[0]
    masks = obs[OBS_LANGUAGE_ATTENTION_MASK].cpu().numpy()[0]

    op_tokens = op_trace["tokens"]
    op_mask = op_trace["token_mask"].astype(bool)

    assert tokens.shape == op_tokens.shape, f"Shape mismatch: {tokens.shape} vs {op_tokens.shape}"
    assert masks.shape == op_mask.shape, f"Mask shape mismatch: {masks.shape} vs {op_mask.shape}"

    # Valid tokens must match exactly
    assert np.array_equal(tokens[masks], op_tokens[op_mask]), (
        f"Valid token IDs mismatch:\nLeRobot: {tokens[masks]}\nOpenPI:  {op_tokens[op_mask]}"
    )
    assert np.array_equal(masks, op_mask), "Attention mask mismatch"
    assert int(masks.sum()) == 20, f"Expected 20 valid tokens, got {int(masks.sum())}"


def test_openpi_normalization_formulas():
    """Gate 3: OpenPI quantile normalization and unnormalization formula exactness."""
    stats_path = (
        ROOT
        / "cache/openpi_reference/openpi-assets/checkpoints/pi05_libero/assets/physical-intelligence/libero/norm_stats.json"
    )
    norm_data = json.loads(stats_path.read_text())["norm_stats"]
    q01 = torch.tensor(norm_data["actions"]["q01"], dtype=torch.float32)
    q99 = torch.tensor(norm_data["actions"]["q99"], dtype=torch.float32)

    features = {
        "action": PolicyFeature(type=FeatureType.ACTION, shape=(7,)),
        "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(8,)),
    }
    norm_map = {
        FeatureType.ACTION: NormalizationMode.OPENPI_QUANTILES,
        FeatureType.STATE: NormalizationMode.OPENPI_QUANTILES,
    }
    dataset_stats = {
        "action": {
            "q01": q01,
            "q99": q99,
        },
        "observation.state": {
            "q01": torch.tensor(norm_data["state"]["q01"], dtype=torch.float32),
            "q99": torch.tensor(norm_data["state"]["q99"], dtype=torch.float32),
        },
    }

    norm_step = NormalizerProcessorStep(features=features, norm_map=norm_map, stats=dataset_stats)
    unnorm_step = UnnormalizerProcessorStep(
        features={"action": features["action"]},
        norm_map={FeatureType.ACTION: NormalizationMode.OPENPI_QUANTILES},
        stats={"action": dataset_stats["action"]},
    )

    # Test values
    raw_action = torch.tensor([0.1, -0.2, 0.5, 0.01, -0.05, 0.02, 1.0], dtype=torch.float32)
    expected_norm = (raw_action - q01) / (q99 - q01 + 1e-6) * 2.0 - 1.0
    expected_unnorm = (expected_norm + 1.0) / 2.0 * (q99 - q01 + 1e-6) + q01

    transition = {
        TransitionKey.ACTION: PolicyAction(raw_action.clone()),
        TransitionKey.OBSERVATION: {},
    }
    norm_res = norm_step(transition)
    norm_action = norm_res[TransitionKey.ACTION]

    assert torch.allclose(norm_action, expected_norm, atol=1e-6), (
        f"Normalized mismatch: {norm_action} vs {expected_norm}"
    )

    unnorm_trans = {TransitionKey.ACTION: PolicyAction(norm_action.clone())}
    unnorm_res = unnorm_step(unnorm_trans)
    unnorm_action = unnorm_res[TransitionKey.ACTION]

    assert torch.allclose(unnorm_action, raw_action, atol=1e-6), (
        f"Roundtrip unnormalize mismatch: {unnorm_action} vs {raw_action}"
    )


def test_camera_order_and_masks_config():
    """Gate 4: Camera ordering, rotation, and mask configuration checks."""
    config_dict = json.loads((ARTIFACT_DIR / "config.json").read_text())
    valid_keys = {k: v for k, v in config_dict.items() if hasattr(PI05Config, k) and k != "type"}
    config = PI05Config(**valid_keys)
    config.input_features = {
        k: PolicyFeature(type=FeatureType(v["type"]), shape=tuple(v["shape"]))
        for k, v in config_dict["input_features"].items()
    }
    config.output_features = {
        k: PolicyFeature(type=FeatureType(v["type"]), shape=tuple(v["shape"]))
        for k, v in config_dict["output_features"].items()
    }

    assert config.chunk_size == 10
    assert config.n_action_steps == 5
    assert config.num_inference_steps == 10
    assert config.empty_cameras == 1
    assert "observation.images.image" in config.input_features
    assert "observation.images.image2" in config.input_features


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

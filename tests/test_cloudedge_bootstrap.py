#!/usr/bin/env python3
"""Unit tests for CloudEdge pi05 bootstrap configuration, tokenization, normalization, and alignment."""

import json
from pathlib import Path
import numpy as np
import pytest
import torch

from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
from lerobot.processor import (
    PaliGemmaTokenizerProcessorStep,
    TransitionKey,
)
from lerobot.utils.constants import (
    OBS_LANGUAGE_ATTENTION_MASK,
    OBS_LANGUAGE_TOKENS,
)
from lerobot_policy_cloudedge_pi05.configuration_cloudedge_pi05 import CloudEdgePI05Config
from lerobot_policy_cloudedge_pi05.processor_cloudedge_pi05 import (
    SelectCurrentStateProcessorStep,
)

ROOT = Path(__file__).resolve().parents[1]
ALIGNED_BASE_DIR = ROOT / "artifacts/pi05_libero_base_official_aligned"
BOOTSTRAP_DIR = ROOT / "artifacts/cloudedge_pi05_v2_bootstrap_official_aligned"


def test_bootstrap_config_inheritance():
    """Verify CloudEdge config inherits aligned base contract."""
    base_config = json.loads((ALIGNED_BASE_DIR / "config.json").read_text())

    valid_fields = {k: v for k, v in base_config.items() if hasattr(CloudEdgePI05Config, k) and k != "type"}
    valid_fields.update(
        {
            "architecture_version": 2,
            "freeze_vision_encoder": True,
            "history_window": 21,
            "stale_loss_weight_max": 0.5,
            "stale_loss_warmup_steps": 10000,
            "eval_delay_max": 0,
            "cloudedge_train_step": 0,
        }
    )
    config = CloudEdgePI05Config(**valid_fields)

    assert config.architecture_version == 2
    assert config.freeze_vision_encoder is True
    assert config.cloudedge_train_step == 0
    assert config.chunk_size == 10, f"Expected chunk_size 10, got {config.chunk_size}"
    assert config.n_action_steps == 5, f"Expected n_action_steps 5, got {config.n_action_steps}"
    assert config.num_inference_steps == 10, f"Expected num_inference_steps 10, got {config.num_inference_steps}"
    assert config.max_action_dim == 32
    assert config.max_state_dim == 32
    assert config.empty_cameras == 1
    assert config.history_window == 21
    assert config.stale_loss_weight_max == 0.5
    assert config.stale_loss_warmup_steps == 10000


def test_select_current_state_processor():
    """Verify SelectCurrentStateProcessorStep retains current state at t."""
    step = SelectCurrentStateProcessorStep()

    # Batched case: (B, T, D)
    b, t, d = 2, 21, 8
    history_state = torch.arange(b * t * d, dtype=torch.float32).view(b, t, d)
    transition = {
        TransitionKey.OBSERVATION: {
            "observation.state": history_state,
        }
    }
    result = step(transition)
    curr_state = result[TransitionKey.OBSERVATION]["observation.state"]
    assert curr_state.shape == (b, d)
    assert torch.equal(curr_state, history_state[:, -1])

    # Unbatched case: (T, D)
    history_unbatched = torch.arange(t * d, dtype=torch.float32).view(t, d)
    transition_unbatched = {
        TransitionKey.OBSERVATION: {
            "observation.state": history_unbatched,
        }
    }
    res_unbatched = step(transition_unbatched)
    curr_unbatched = res_unbatched[TransitionKey.OBSERVATION]["observation.state"]
    assert curr_unbatched.shape == (d,)
    assert torch.equal(curr_unbatched, history_unbatched[-1])


def test_cloudedge_token_and_norm_alignment():
    """Verify CloudEdge tokenizer and normalization produce exact 20 tokens and OpenPI quantiles."""
    tok_path = ALIGNED_BASE_DIR / "paligemma_tokenizer.model"
    assert tok_path.exists(), f"Missing {tok_path}"

    step = PaliGemmaTokenizerProcessorStep(
        tokenizer_path=str(tok_path),
        max_length=200,
        task_key="task",
        clean_text=True,
    )
    prompt = "pick up the black bowl between the plate and the ramekin and place it on the plate"
    transition = {
        TransitionKey.OBSERVATION: {},
        TransitionKey.COMPLEMENTARY_DATA: {"task": prompt},
    }
    step._current_transition = transition
    obs = step.observation({})

    tokens = obs[OBS_LANGUAGE_TOKENS].cpu().numpy()[0]
    masks = obs[OBS_LANGUAGE_ATTENTION_MASK].cpu().numpy()[0]

    assert int(masks.sum()) == 20
    expected_ids = np.array(
        [2, 18075, 908, 573, 2656, 14581, 1865, 573, 8811, 578, 573, 117064, 4074, 578, 2040, 665, 611, 573, 8811, 108]
    )
    assert np.array_equal(tokens[:20], expected_ids)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

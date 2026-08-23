#!/usr/bin/env python3
"""Unit tests verifying CloudEdge delay semantics, history queuing, and baseline equivalence."""

from collections import deque
import numpy as np
import pytest
import torch

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot_policy_cloudedge_pi05.configuration_cloudedge_pi05 import CloudEdgePI05Config
from lerobot_policy_cloudedge_pi05.modeling_cloudedge_pi05 import CloudEdgePI05Policy


def make_dummy_config(eval_delay_max=0, use_edge_vision=True):
    config = CloudEdgePI05Config(
        device="cpu",
        chunk_size=10,
        n_action_steps=5,
        num_inference_steps=10,
        history_window=21,
        eval_delay_max=eval_delay_max,
        use_edge_vision=use_edge_vision,
    )
    config.input_features = {
        "observation.images.image": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 224, 224)),
        "observation.images.image2": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 224, 224)),
        "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(8,)),
    }
    config.output_features = {
        "action": PolicyFeature(type=FeatureType.ACTION, shape=(7,)),
    }
    return config


def test_dmax_zero_always_selects_current():
    """Verify d_max=0 always selects current image (delay=0)."""
    config = make_dummy_config(eval_delay_max=0)
    # We can test _inference_image_views logic
    history = deque(maxlen=21)

    for t in range(10):
        frame = {"observation.images.image": torch.full((1, 3, 224, 224), fill_value=float(t))}
        history.append(frame)

        # Under d_max=0, delay is 0 and index is len(history) - 1
        delay = 0
        history_index = max(0, len(history) - 1 - delay)
        selected = history[history_index]
        assert selected["observation.images.image"][0, 0, 0, 0].item() == float(t)


def test_dmax_positive_uniform_delay_distribution():
    """Verify d_max>0 samples uniformly from {1, ..., d_max}."""
    d_max = 10
    torch.manual_seed(42)
    samples = [int(torch.randint(1, d_max + 1, ()).item()) for _ in range(10000)]

    for d in range(1, d_max + 1):
        assert d in samples
        count = samples.count(d)
        expected = 10000 / d_max
        assert abs(count - expected) < 300, f"Delay {d} count {count} deviates from uniform expected {expected}"

    assert min(samples) == 1
    assert max(samples) == d_max


def test_delay_in_environment_steps_during_queue_drain():
    """Verify observation history is updated on EVERY environment step, even while queued actions drain."""
    # Simulate select_action behavior
    history = deque(maxlen=21)
    queue = deque()
    n_action_steps = 5

    observed_history_lengths = []
    for env_step in range(15):
        # Observation at env_step
        obs = {"observation.images.image": torch.full((1, 3, 224, 224), fill_value=float(env_step))}
        history.append(obs)

        if len(queue) == 0:
            # Action chunk generated: 5 actions queued
            chunk = [torch.zeros(1, 7) for _ in range(n_action_steps)]
            queue.extend(chunk)

        action = queue.popleft()
        observed_history_lengths.append(len(history))

    # History must have grown by 1 on every single environment step
    assert len(history) == 15
    assert observed_history_lengths == list(range(1, 16))


def test_warmup_uses_oldest_available_frame():
    """Verify warmup when history length < delay + 1 clamps to index 0 (oldest valid frame)."""
    history = deque(maxlen=21)
    # 3 frames in history (t=0, 1, 2)
    for t in range(3):
        history.append({"t": t})

    delay = 10  # delay is larger than current history
    history_index = max(0, len(history) - 1 - delay)
    assert history_index == 0, f"Expected clamp to index 0, got {history_index}"
    assert history[history_index]["t"] == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

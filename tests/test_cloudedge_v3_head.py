#!/usr/bin/env python3
"""Unit tests for the post-flow CloudEdge pi0.5 v3 correction contract."""

import re

import pytest
import torch
import torch.nn.functional as F

from lerobot_policy_cloudedge_pi05.configuration_cloudedge_pi05_v3 import (
    CloudEdgePI05V3Config,
)
from lerobot_policy_cloudedge_pi05.modeling_cloudedge_pi05_v3 import (
    CloudEdgePI05V3Policy,
    CloudPlan,
    EdgeActionCorrectionHead,
)


def test_correction_head_preserves_base_then_opens_edge_gradient_path():
    torch.manual_seed(7)
    head = EdgeActionCorrectionHead(edge_width=8, expert_width=16, action_dim=4)
    base = torch.randn(3, 5, 4)
    cloud = torch.randn(3, 5, 16)
    edge = torch.randn(3, 8)
    target = torch.randn(3, 5, 4)

    initial = head(cloud, edge, base)
    assert torch.count_nonzero(initial).item() == 0
    assert torch.count_nonzero(head.edge_projection.weight).item() > 0
    assert torch.count_nonzero(head.fusion[-1].weight).item() == 0

    optimizer = torch.optim.AdamW(head.parameters(), lr=1e-2)
    optimizer.zero_grad()
    F.l1_loss(base + head(cloud, edge, base), target).backward()
    assert torch.count_nonzero(head.fusion[-1].weight.grad).item() > 0
    optimizer.step()

    optimizer.zero_grad()
    F.l1_loss(base + head(cloud, edge, base), target).backward()
    assert torch.count_nonzero(head.edge_projection.weight.grad).item() > 0


def test_cloud_plan_is_reusable_with_new_edge_observation():
    torch.manual_seed(8)
    head = EdgeActionCorrectionHead(edge_width=8, expert_width=16, action_dim=4)
    with torch.no_grad():
        head.fusion[-1].weight.normal_(std=0.01)
    plan = CloudPlan(
        actions=torch.randn(2, 5, 4),
        action_features=torch.randn(2, 5, 16),
    )
    edge_a = torch.randn(2, 8)
    edge_b = torch.randn(2, 8)
    corrected_a = plan.actions + head(plan.action_features, edge_a, plan.actions)
    corrected_b = plan.actions + head(plan.action_features, edge_b, plan.actions)
    assert not torch.equal(corrected_a, corrected_b)
    assert torch.equal(plan.actions, plan.actions.clone())


def test_peft_targets_cover_language_and_expert_only():
    policy = object.__new__(CloudEdgePI05V3Policy)
    config = policy._get_default_peft_targets()
    pattern = re.compile(config["target_modules"])
    expected = [
        "model.paligemma_with_expert.paligemma.model.language_model.layers.0.self_attn.q_proj",
        "model.paligemma_with_expert.paligemma.model.language_model.layers.17.mlp.down_proj",
        "model.paligemma_with_expert.gemma_expert.model.layers.0.self_attn.v_proj",
        "model.paligemma_with_expert.gemma_expert.model.layers.17.mlp.gate_proj",
    ]
    excluded = [
        "model.paligemma_with_expert.paligemma.model.vision_tower.vision_model.encoder.layers.0.self_attn.q_proj",
        "model.edge_vision.vision_model.encoder.layers.0.self_attn.q_proj",
        "model.action_in_proj",
        "model.action_out_proj",
        "model.time_mlp_in",
        "model.time_mlp_out",
    ]
    assert all(pattern.fullmatch(name) for name in expected)
    assert not any(pattern.fullmatch(name) for name in excluded)
    assert config["modules_to_save"] == ["edge_action_head"]
    assert config["r"] == 8


def test_v3_requires_per_step_execution():
    CloudEdgePI05V3Config(n_action_steps=1)
    with pytest.raises(ValueError, match="n_action_steps=1"):
        CloudEdgePI05V3Config(n_action_steps=5)

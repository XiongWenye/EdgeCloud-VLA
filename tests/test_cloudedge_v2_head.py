#!/usr/bin/env python3
"""Unit tests for the v2 residual edge action head and PEFT contract."""

import re

import torch
import torch.nn.functional as F

from lerobot_policy_cloudedge_pi05.modeling_cloudedge_pi05 import (
    CloudEdgePI05Policy,
    EdgeResidualActionHead,
)


def test_residual_head_preserves_base_then_opens_gradient_path():
    torch.manual_seed(7)
    head = EdgeResidualActionHead(edge_width=8, expert_width=16, action_dim=4)

    cloud = torch.randn(3, 5, 16)
    edge = torch.randn(3, 8)
    target = torch.randn(3, 5, 4)

    initial = head(cloud, edge)
    assert torch.count_nonzero(initial).item() == 0
    assert torch.count_nonzero(head.edge_projection.weight).item() > 0
    assert torch.count_nonzero(head.fusion[-1].weight).item() == 0

    optimizer = torch.optim.AdamW(head.parameters(), lr=1e-2)
    optimizer.zero_grad()
    F.mse_loss(head(cloud, edge), target).backward()

    final_grad = head.fusion[-1].weight.grad
    assert final_grad is not None
    assert torch.count_nonzero(final_grad).item() > 0
    optimizer.step()

    optimizer.zero_grad()
    F.mse_loss(head(cloud, edge), target).backward()

    edge_grad = head.edge_projection.weight.grad
    assert edge_grad is not None
    assert torch.count_nonzero(edge_grad).item() > 0


def test_peft_targets_cover_planning_layers_but_not_vision():
    policy = object.__new__(CloudEdgePI05Policy)
    config = policy._get_default_peft_targets()
    pattern = re.compile(config["target_modules"])

    expected = [
        "model.paligemma_with_expert.paligemma.model.language_model.layers.0.self_attn.q_proj",
        "model.paligemma_with_expert.paligemma.model.language_model.layers.17.mlp.down_proj",
        "model.paligemma_with_expert.gemma_expert.model.layers.0.self_attn.v_proj",
        "model.paligemma_with_expert.gemma_expert.model.layers.17.mlp.gate_proj",
    ]
    excluded = [
        "model.paligemma_with_expert.paligemma.model.vision_tower."
        "vision_model.encoder.layers.0.self_attn.q_proj",
        "model.edge_vision.vision_model.encoder.layers.0.self_attn.q_proj",
        "model.action_out_proj",
    ]

    assert all(pattern.fullmatch(name) for name in expected)
    assert not any(pattern.fullmatch(name) for name in excluded)
    assert config["modules_to_save"] == [
        "action_in_proj",
        "action_out_proj",
        "time_mlp_in",
        "time_mlp_out",
        "edge_action_head",
    ]

# CloudEdge pi0.5 v2 design

## Objective

Recover the aligned pi0.5 LIBERO policy's synchronous behavior while learning the
paper's current-edge / stale-cloud specialization under a 21-frame observation
window. A long run is not admissible until small experiments prove both base
retention and non-trivial use of the current edge image.

## Paper invariants retained

- The cloud path receives either the current observation or one episode-safe
  stale observation sampled uniformly from offsets 1..20.
- The edge path always receives the current observation.
- Both forward passes use the same language, action target, flow noise, and flow
  time.
- The objective remains exactly
  `(1 - lambda) * L_fresh + lambda * L_stale`; no auxiliary distillation or
  consistency loss is introduced in the first faithful reproduction.
- Lambda follows an explicit curriculum from zero to `lambda_max`.
- The edge SigLIP-Base encoder is frozen. The PaliGemma SigLIP tower is also
  frozen to protect the aligned visual representation.
- Cloud/planning weights are adapted with LoRA and the new action head is
  trainable, matching the paper's LoRA cloud VLA plus trainable edge action head.

## pi0.5-specific edge action head

The paper fuses cloud planning features with current edge visual features in a
residual action head. For flow matching the corresponding planning feature is
the action-expert hidden state at every noisy-action token and denoising time.
The v2 head therefore computes:

```
edge = Linear(LayerNorm(mean_views(frozen_edge_siglip(current_images))))
delta_v = MLP(concat(LayerNorm(action_expert_hidden), edge))
velocity = pretrained_action_out(action_expert_hidden) + delta_v
```

The final `delta_v` projection is zero-initialized, but the upstream edge
projection is normally initialized. This preserves the pretrained flow field
exactly at step zero without creating the v1 all-zero/two-layer gradient trap.
The head is applied identically during training and at every ODE denoising step
during inference.

The new module owns a persistent `train_step` buffer so the lambda curriculum
survives PEFT checkpoint save/load and resume.

## Trainable parameter policy

- frozen: PaliGemma vision tower; edge SigLIP vision tower;
- LoRA rank 16: all q/k/v/o and gate/up/down projections in the PaliGemma
  language model and the 300M action expert;
- fully trained and saved: action input/output projections, time MLP, edge
  residual action head;
- no unrestricted full update of the 2B PaliGemma language model.

This is intentionally different from v1, which fully updated the cloud
language model and action expert at batch 8 while the edge vision branch was
mathematically dead.

## Required gates

1. **Static/base-parity gate**: v2 bootstrap loads all aligned base tensors;
   `delta_v == 0` at initialization; the base and v2 action chunks match for
   fixed observations and noise within numerical tolerance.
2. **Gradient gate**: first backward pass gives a non-zero gradient to the final
   residual projection; after one optimizer step, a second backward pass gives
   non-zero gradient to the edge projection. Both vision towers have no
   gradients.
3. **PEFT gate**: printed trainable names contain only LoRA adapters, the five
   designated action/time modules, and the edge head. Save/load preserves
   outputs and `train_step`.
4. **Short-run gate**: compare 5k/10k candidates at LR 5e-6 and 1e-5. Evaluate
   LIBERO Spatial at d=0 and d=10 with the same seed and fixed episode count.
5. **Admission gate for long training**: synchronous smoke performance must
   retain at least 85% absolute success and d=10 must not be below the aligned
   base smoke result. Edge-on must measurably change the velocity/action under
   stale cloud input; otherwise the run is rejected.
6. **Final gate**: full four-suite evaluation at d_max in {0,10,20,40}, at least
   three seeds for CloudEdge and matched baseline seeds, plus an edge-off
   ablation and confidence intervals.

## Optimization policy

Do not repeat the v1 120k run. Candidate selection is based on closed-loop
performance, not training loss. Start from the aligned official checkpoint,
use batch size as large as H200 memory permits, AdamW with gradient clipping,
and preserve the official 10-action prediction horizon / 5-action execution
horizon. Only the winning short-run configuration advances to a longer
checkpoint schedule.

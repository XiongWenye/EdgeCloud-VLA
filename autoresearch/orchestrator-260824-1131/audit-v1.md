# CloudEdge pi05 v1 failure audit

## Incumbent

- Source commit: cbaade9036481f12293d2a185bacaacdd50642de
- Final checkpoint: outputs/cloudedge_pi05_libero_aligned_120k/checkpoints/120000/pretrained_model
- Training job: 41490778, resumed from the complete 55k checkpoint and completed at 120k
- Evaluation grid: 8 delays x 4 suites x 3 CloudEdge seeds, plus one delayed-base seed

## Measured failure

- aligned delayed base d0 macro: 96.25%
- v1 CloudEdge d0 macro: 30.48% +/- 1.77%
- v1 CloudEdge d10 macro: 46.33% +/- 1.80%
- v1 CloudEdge d20 macro: 23.28% +/- 1.91%
- v1 CloudEdge d40 macro: 4.38% +/- 0.71%
- full-curve AUC: 25.12%, below delayed base 25.57%

The d0 failure is stable across seeds 7, 8, and 9, so it is not rollout noise.

## Proven root cause: dead edge projection

scripts/bootstrap_checkpoint.py zeroes every parameter in edge_projection after
the model constructor has correctly zeroed only the last layer. With the
two-layer MLP Linear -> SiLU -> Linear, zeroing both weights makes the hidden
activation exactly zero and prevents either weight matrix from ever receiving a
gradient. Only the final bias can train.

Direct safetensors evidence:

| Tensor | Bootstrap norm | 120k norm |
| --- | ---: | ---: |
| edge_projection.0.weight | 0.0 | 0.0 |
| edge_projection.0.bias | 0.0 | 0.0 |
| edge_projection.2.weight | 0.0 | 0.0 |
| edge_projection.2.bias | 0.0 | 0.076524 |

The final model therefore cannot condition actions on the current edge image.
The mechanism diagnostic independently confirms that edge-on and edge-off
predictions differ by approximately zero at every positive delay.

## Contributing design mismatch: unconstrained backbone updates

The paper uses a LoRA-adapted cloud VLA and a trainable edge action head. V1
instead uses use_peft=false and train_expert_only=false, so all non-vision
PaliGemma parameters and the full action expert update for 120k optimizer steps.
This permits catastrophic forgetting of the already aligned 96.9% pi05 policy.

The official OpenPI pi05_libero recipe full-finetunes for 30k steps with global
batch 256, LR 5e-5, and EMA 0.999. V1 uses 120k steps, global batch 8, LR
2.5e-5, and no EMA. Matching optimizer steps while changing batch by 32x is not
an equivalent optimization budget.

## Weak fusion

V1 adds one image vector identically to every noisy-action token before the
Gemma action expert. This is weaker than the paper's residual vision-augmented
action head and can be attenuated by expert normalization. V2 must inject edge
conditioning through both a token residual and the pi05 AdaRMS conditioning
path while preserving exact base behavior at initialization.

## Diagnostic-report defect

The mechanism report labels an inference-time edge toggle as a trained no-edge
ablation and claims significant improvement although measured positive-delay
reductions are approximately zero. The d0 edge RMSE is zero by construction
because the reference is the same prediction. These claims must not be reused.

## V2 acceptance gates

1. Bootstrap fixed-noise d0 output equals aligned base.
2. Edge fusion starts at zero output, but its last-layer and upstream gradients
   become nonzero after optimization begins.
3. Frozen edge encoder has no gradients.
4. Cloud base weights are frozen except explicit LoRA adapters; the pi05 action
   expert and new edge fusion remain trainable.
5. A short checkpoint sweep retains at least 85% d0 macro on a held-out smoke
   grid before any 120k submission.
6. Edge-on measurably reduces stale-cloud action drift relative to edge-off.
7. Only a candidate passing d0 and edge-use gates advances to full training.

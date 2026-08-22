# CloudEdgeVLA protocol extracted from arXiv:2608.00569v1

## Method

At environment step t, the cloud backbone receives image `o[t-k]` and language, while the edge receives current image `o[t]`. The edge never blocks for a specific cloud response. In training, a window of W consecutive frames supplies current `o[t]` and delayed `o[t-d]`, where `d ~ Uniform(1, W-1)`. Both cloud paths are fused with the same current edge image and supervised against the same current action chunk. The stale branch is not detached. A curriculum increases the stale loss coefficient from zero to λmax.

For the π0.5 port, action L1 is replaced by conditional flow matching. For action chunk `a`, noise `ε`, and sampled time `τ`, both paths receive the same `x_τ = τ ε + (1-τ)a` and target velocity `u_τ = ε-a`. This shared stochastic target is important: otherwise noise variance would confound the fresh/stale comparison.

## Paper experiments

- Benchmark: LIBERO Spatial, Object, Goal, and Long, 10 tasks/suite, 50 demonstrations/task.
- Closed-loop conditions: no delay and `k ~ Uniform({1,...,d_max})` for `d_max = 5,10,15,20,25,30,40`.
- Reported main table subset: `d_max = 0,10,20,40`.
- Trials: 50/task. CloudEdgeVLA uses seeds 7, 8, and 9; the table reports mean and standard deviation across seeds.
- Training diagnostic window: W=21, inferred directly from the supplement's statement that fixed offsets through 20 lie within the paired-frame window.
- Mechanism diagnostic: 80 LIBERO-Spatial states, eight states from one demonstration in each of 10 tasks, fixed delays `0,1,3,5,8,10,15,20`.
- Reported aligned CloudEdge checkpoint: 120k steps.
- Offline action diagnostic: normalized 8-step, 7-dimensional chunks; current proprioception for every cloud query.
- Ablations at `d_max=10`: no edge vision; frozen SigLIP-Base; frozen SigLIP-SO400M; fresh-only; stale-only; joint loss.
- Real-robot pilot: Franka toy-bear-to-box, static/dynamic target, 10 trials/cell, additional RTT 0/400/1000 ms.

## Paper results used as references, not π0.5 targets

At `d_max=40`, CloudEdgeVLA reports Spatial 76.4%, Object 75.6%, Goal 78.0%, and Long 63.8%. Its four-suite macro delay AUC is 90.8%, and its retention at 40 steps is 76.5%. These are OpenVLA-OFT-based results and must not be relabeled as π0.5 results.

## Missing details and declared choices

The paper does not disclose λmax, curriculum length, optimizer, learning rate, batch size, LoRA rank/targets, precise action execution horizon in the main evaluation, dataset artifact revision, or checkpoint artifact. This repository therefore declares every substituted value in `configs/experiment.json`. Any reported π0.5 result must include that manifest and the exact Git commit.


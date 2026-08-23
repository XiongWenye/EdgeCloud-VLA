# CloudEdge π0.5 on LIBERO (Aligned Reproduction)

This repository reproduces the CloudEdgeVLA paired-frame method on top of the officially aligned LeRobot π0.5 flow-matching policy (`artifacts/pi05_libero_base_official_aligned`, verified at **96.90%** LIBERO macro benchmark).

The adaptation operates at the conditional flow field:
- A cloud π0.5 prefix encodes current or delayed visual observations and language into per-layer KV context.
- A frozen SigLIP-Base encoder (`google/siglip-base-patch16-224`) processes current edge vision at time $t$.
- A trainable zero-initialized projection adds edge visual features to every noisy action token in the π0.5 action expert.
- Dual-path training applies paired flow-matching supervision with shared noise and flow timestep:
  $$\mathcal{L} = (1 - \lambda) \text{MSE}(v_{\text{fresh}}, u_t) + \lambda \text{MSE}(v_{\text{stale}}, u_t)$$
- Current proprioception and text prompt are never delayed.

## Verified Reproduction Contract

- **Base Policy**: `artifacts/pi05_libero_base_official_aligned` (Spatial: 98.60%, Object: 99.00%, Goal: 98.20%, LIBERO-10: 91.80%, Macro: **96.90%**)
- **Inference Protocol**: 10 action predictions, execute 5 actions before replanning (`chunk_size=10`, `n_action_steps=5`, `num_inference_steps=10`)
- **Normalization**: `OPENPI_QUANTILES` matching Physical Intelligence OpenPI reference formula
- **Language Tokenizer**: Local PaliGemma SentencePiece tokenizer (`paligemma_tokenizer.model`, exact 20 tokens, no state in prompt)
- **Edge Encoder**: Frozen `google/siglip-base-patch16-224@7fd15f0689c79d79e38b1c2e2e2370a7bf2761ed`
- **Training Setup**: 120,000 optimizer steps, 4 $\times$ H200 GPUs, global batch 8, history window 21, uniform stale offset $U(1..20)$, $\lambda_{\max}=0.5$ with 10k-step linear warmup, save interval 5k steps
- **Evaluation Matrix**: 4 suites (Spatial, Object, Goal, LIBERO-10) $\times$ delays $d_{\max} \in \{0, 5, 10, 15, 20, 25, 30, 40\}$ $\times$ 3 CloudEdge seeds (7, 8, 9) + 1 delayed baseline seed (7), 50 episodes/task $\times$ 10 tasks = 500 episodes/suite

## Lineage & Gate Structure

1. **Gate 1**: Deterministic setup & LeRobot alignment patch verification
2. **Gate 2**: CloudEdge preprocessing & aligned bootstrap (`artifacts/cloudedge_pi05_bootstrap_official_aligned`)
3. **Gate 3**: Delay semantics & environment-step history verification
4. **Gate 4**: H200 smoke validation (inference, train step, spatial rollout, delay trace)
5. **Gate 5**: 120k fine-tuning (`outputs/cloudedge_pi05_libero_aligned_120k`)
6. **Gate 6**: 64-record core closed-loop evaluation matrix ($d_{\max} \in \{0, 10, 20, 40\}$)
7. **Gate 7**: Complete 8-point delay curve ($d_{\max} \in \{5, 15, 25, 30\}$)
8. **Gate 8**: Mechanism diagnostics on LIBERO Spatial (80 states, delays 0..20)
9. **Gate 9**: Trained ablations at $d_{\max}=10$
10. **Gate 10**: Aggregation, delay AUC, retention metrics, and final report

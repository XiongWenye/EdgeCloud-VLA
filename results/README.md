# CloudEdge pi0.5 released evaluation results

This directory contains the machine-readable metrics used by
`docs/PI05_CLOUDEDGE_COMPATIBILITY_EVIDENCE.md`.

## Canonical result sets

| Result set | Raw JSON files | Episodes per file | Total episodes | Purpose |
|---|---:|---:|---:|---|
| `baseline_aligned_delayed/raw` | 32 | 500 | 16,000 | Aligned pi0.5 delayed baseline, four suites and eight delay points |
| `cloudedge_pi05_v2_candidates` | 12 | 100 | 1,200 | V2 learning-rate/checkpoint selection |
| `cloudedge_pi05_v2_full/raw` | 48 | 500 | 24,000 | V2 winner, four suites, four delays, three seeds |
| `cloudedge_pi05_v2_edgezero/raw` | 16 | 500 | 8,000 | Seed-matched V2 edge-feature ablation |
| `cloudedge_pi05_v3_candidates` | 12 | 100 | 1,200 | V3 learning-rate/checkpoint gate |
| `cloudedge_pi05_v3_diagnostics` | 4 | 100 | 400 | Horizon-matched V3 bootstrap and edge-zero controls |
| **Total** | **124** | — | **50,800** | — |

Only `eval_info.json` files are released from raw evaluator directories.
The 8,000 generated MP4 rollout videos are intentionally excluded because
they occupy hundreds of megabytes. Absolute `video_paths` inside evaluator
JSON files preserve provenance but do not resolve outside the Harvard cluster.

## Canonical summaries

- `cloudedge_pi05_v2_candidates_summary.{json,md}`
- `cloudedge_pi05_v2_full_summary.{json,md}`
- `cloudedge_pi05_v2_edge_ablation_summary.{json,md}`
- `cloudedge_pi05_v3_candidates_summary.{json,md}`
- `cloudedge_pi05_v3_diagnostics_summary.{json,md}`

## Release audit

Before release, every raw JSON file was parsed and checked for:

- exactly ten LIBERO tasks;
- the expected episode count;
- no non-finite JSON constants;
- exact agreement between the stored success percentage and the recomputed
  count of per-episode success booleans.

All aggregate means, sample standard deviations, edge-ablation differences,
candidate gates, and horizon-matched diagnostics were independently recomputed
from raw JSON and matched the released summaries to an absolute tolerance of
`1e-9`.

The V3 comparisons contain only one seed and 100 episodes per point. Differences
of four to five percentage points should therefore be treated as observed
effect sizes, not as statistically conclusive estimates. The preregistered
candidate-gate failure is much larger and is the basis for the stated
architecture-portability conclusion.

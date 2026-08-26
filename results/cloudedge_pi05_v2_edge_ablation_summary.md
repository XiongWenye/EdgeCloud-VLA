# CloudEdge pi0.5 v2 edge-feature ablation

Winner: lr=5e-6, step=2500. Seed-matched closed-loop evaluation, 500 episodes per suite/point.

Edge-zero keeps the trained action head and replaces only its edge feature with zeros. It does not disable the complete residual head.

| Suite | dmax | Edge on | Edge zero | Edge contribution |
|---|---:|---:|---:|---:|
| libero_spatial | 0 | 95.6% | 94.8% | +0.8 pp |
| libero_spatial | 10 | 38.8% | 39.8% | -1.0 pp |
| libero_spatial | 20 | 4.6% | 2.6% | +2.0 pp |
| libero_spatial | 40 | 0.0% | 0.0% | +0.0 pp |
| libero_object | 0 | 98.4% | 98.2% | +0.2 pp |
| libero_object | 10 | 52.8% | 48.2% | +4.6 pp |
| libero_object | 20 | 3.2% | 1.8% | +1.4 pp |
| libero_object | 40 | 0.0% | 0.0% | +0.0 pp |
| libero_goal | 0 | 91.2% | 90.8% | +0.4 pp |
| libero_goal | 10 | 63.8% | 66.6% | -2.8 pp |
| libero_goal | 20 | 30.6% | 36.4% | -5.8 pp |
| libero_goal | 40 | 9.2% | 6.0% | +3.2 pp |
| libero_10 | 0 | 90.8% | 82.8% | +8.0 pp |
| libero_10 | 10 | 55.0% | 36.8% | +18.2 pp |
| libero_10 | 20 | 11.4% | 4.4% | +7.0 pp |
| libero_10 | 40 | 0.0% | 0.2% | -0.2 pp |

## Four-suite macro average

| dmax | Edge on | Edge zero | Edge contribution |
|---:|---:|---:|---:|
| 0 | 94.00% | 91.65% | +2.35 pp |
| 10 | 52.60% | 47.85% | +4.75 pp |
| 20 | 12.45% | 11.30% | +1.15 pp |
| 40 | 2.30% | 1.55% | +0.75 pp |

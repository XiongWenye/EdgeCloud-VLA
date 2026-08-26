# CloudEdge π0.5 Aligned Full Reproduction Summary

Evaluation across LIBERO benchmarks under simulated observation delay.
Core protocol: 50 episodes/task $\times$ 10 tasks = 500 episodes/suite, chunk 10, replan 5, flow steps 10.

## Macro Summary & Metrics

| Metric | Delayed Baseline (Seed 7) | **CloudEdge Aligned (Seeds 7, 8, 9)** | Paper Reference (OpenVLA-OFT) |
|---|---|---|---|
| Macro Success ($d_{max} = 0$) | 96.25% | **93.83% ± 0.15%** | - |
| Macro Success ($d_{max} = 10$) | 41.90% | **53.93% ± 1.63%** | - |
| Macro Success ($d_{max} = 20$) | 8.20% | **12.50% ± 0.18%** | - |
| Macro Success ($d_{max} = 40$) | 2.10% | **2.35% ± 0.48%** | - |
| **Delay AUC (0 → 40)** | 26.11% | **30.49%** | 90.8% |
| **Retention at $d_{max}=40$** | 2.18% | **2.50%** | 76.5% |

## Per-Suite Success Rates

| Benchmark Suite | Method | $d_{max}=0$ | $d_{max}=10$ | $d_{max}=20$ | $d_{max}=40$ | AUC | Retention (40) |
|---|---|---|---|---|---|---|---|
| LIBERO-Spatial | Delayed Baseline | 98.4% | 30.0% | 2.8% | 0.2% | 20.9% | 0.2% |
| LIBERO-Spatial | **CloudEdge (Ours)** | **95.5±0.2%** | **41.3±2.2%** | **4.5±0.1%** | **0.1±0.1%** | **24.0%** | **0.1%** |
| LIBERO-Object | Delayed Baseline | 99.0% | 39.8% | 1.2% | 0.0% | 22.8% | 0.0% |
| LIBERO-Object | **CloudEdge (Ours)** | **97.5±0.8%** | **54.3±5.5%** | **3.5±0.3%** | **0.0±0.0%** | **27.1%** | **0.0%** |
| LIBERO-Goal | Delayed Baseline | 96.2% | 59.6% | 21.4% | 8.2% | 37.0% | 8.5% |
| LIBERO-Goal | **CloudEdge (Ours)** | **92.6±1.2%** | **63.8±0.2%** | **31.7±1.1%** | **9.1±2.1%** | **41.7%** | **9.9%** |
| LIBERO-10 | Delayed Baseline | 91.4% | 38.2% | 7.4% | 0.0% | 23.8% | 0.0% |
| LIBERO-10 | **CloudEdge (Ours)** | **89.8±1.0%** | **56.3±1.7%** | **10.4±0.9%** | **0.1±0.2%** | **29.2%** | **0.1%** |

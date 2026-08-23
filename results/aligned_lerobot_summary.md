# Aligned LeRobot pi05_libero Evaluation Summary

Full 4-suite closed-loop evaluation on Harvard Kempner H200 SLURM cluster (50 episodes/task $	imes$ 10 tasks/suite = 2,000 episodes total, seed 7, wait steps 10, chunk 10, replan 5).

| Benchmark Suite | Published OpenPI | Cluster OpenPI Ref | Unaligned LeRobot | **Aligned LeRobot (Ours)** | Successes | Status |
|---|---|---|---|---|---|---|
| LIBERO-Spatial | 98.8% | 98.4% | 95.2% | **98.60%** | 493/500 | ✅ Completed |
| LIBERO-Object | 98.2% | 99.0% | 96.0% | **99.00%** | 495/500 | ✅ Completed |
| LIBERO-Goal | 98.0% | 97.8% | 94.4% | **98.20%** | 491/500 | ✅ Completed |
| LIBERO-10 (Long Horizon) | 92.4% | 93.4% | 84.4% | **91.80%** | 459/500 | ✅ Completed |
| **Macro Average** | **96.85%** | **97.15%** | **92.50%** | **96.90%** | **1938/2000** | **All 4 Suites** |

## Per-Task Success Breakdown

### LIBERO-Spatial (98.60%)

| Task ID | Task Success Rate | Episodes |
|---|---|---|
| Task 0 | 100.0% | 50 |
| Task 1 | 100.0% | 50 |
| Task 2 | 100.0% | 50 |
| Task 3 | 100.0% | 50 |
| Task 4 | 96.0% | 50 |
| Task 5 | 96.0% | 50 |
| Task 6 | 98.0% | 50 |
| Task 7 | 100.0% | 50 |
| Task 8 | 100.0% | 50 |
| Task 9 | 96.0% | 50 |

### LIBERO-Object (99.00%)

| Task ID | Task Success Rate | Episodes |
|---|---|---|
| Task 0 | 100.0% | 50 |
| Task 1 | 100.0% | 50 |
| Task 2 | 98.0% | 50 |
| Task 3 | 100.0% | 50 |
| Task 4 | 98.0% | 50 |
| Task 5 | 100.0% | 50 |
| Task 6 | 100.0% | 50 |
| Task 7 | 98.0% | 50 |
| Task 8 | 98.0% | 50 |
| Task 9 | 98.0% | 50 |

### LIBERO-Goal (98.20%)

| Task ID | Task Success Rate | Episodes |
|---|---|---|
| Task 0 | 98.0% | 50 |
| Task 1 | 100.0% | 50 |
| Task 2 | 94.0% | 50 |
| Task 3 | 98.0% | 50 |
| Task 4 | 98.0% | 50 |
| Task 5 | 96.0% | 50 |
| Task 6 | 98.0% | 50 |
| Task 7 | 100.0% | 50 |
| Task 8 | 100.0% | 50 |
| Task 9 | 100.0% | 50 |

### LIBERO-10 (Long Horizon) (91.80%)

| Task ID | Task Success Rate | Episodes |
|---|---|---|
| Task 0 | 94.0% | 50 |
| Task 1 | 96.0% | 50 |
| Task 2 | 96.0% | 50 |
| Task 3 | 96.0% | 50 |
| Task 4 | 98.0% | 50 |
| Task 5 | 98.0% | 50 |
| Task 6 | 94.0% | 50 |
| Task 7 | 100.0% | 50 |
| Task 8 | 56.0% | 50 |
| Task 9 | 90.0% | 50 |


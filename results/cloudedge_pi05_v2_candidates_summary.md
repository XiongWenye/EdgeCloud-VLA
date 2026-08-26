# CloudEdge pi0.5 v2 candidate gate

LIBERO Spatial, seed 7, 10 episodes per task (100 episodes per point).

Admission: d0 >= 85.0% and d10 >= 30.0%.

| Candidate | Step | d0 | d10 | Admitted |
|---|---:|---:|---:|:---:|
| lr5e-6 | 2,500 | 97.0% | 45.0% | yes |
| lr5e-6 | 5,000 | 72.0% | 45.0% | no |
| lr5e-6 | 10,000 | 51.0% | 54.0% | no |
| lr1e-5 | 2,500 | 88.0% | 37.0% | yes |
| lr1e-5 | 5,000 | 53.0% | 40.0% | no |
| lr1e-5 | 10,000 | 45.0% | 39.0% | no |

Complete: **True**
Winner: **lr5e-6 step 2500**

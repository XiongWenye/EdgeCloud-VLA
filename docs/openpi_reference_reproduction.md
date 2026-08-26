# OpenPI pi05 LIBERO reference reproduction

This experiment is isolated from the LeRobot conversion and uses the upstream
OpenPI policy server and LIBERO evaluator.

- OpenPI commit: `15a9616a00943ada6c20a0f158e3adb39df2ccac`
- LIBERO submodule: `f78abd68ee283de9f9be3c8f7e2a9ad60246e95c`
- Checkpoint: `gs://openpi-assets/checkpoints/pi05_libero/`
- Checkpoint payload: 16 objects, 12,439,085,481 bytes
- Model action horizon: 10
- Flow-matching Euler steps: 10
- Executed actions before replanning: 5
- Trials: 50 fixed initial states for each of 10 tasks per suite
- Seed: 7
- Wait steps: 10
- Image preprocessing: rotate both cameras 180 degrees, resize with padding to
  224 by 224, and convert to uint8
- Suite limits after the wait period: spatial 220, object 280, goal 300,
  LIBERO-10 520

The Python 3.11 server environment is installed from the upstream `uv.lock`.
Because the historical `rerun-sdk==0.23.1` visualization wheel is no longer
served for Python 3.11, it is omitted. It is not imported by the OpenPI policy
server and does not affect model loading or inference. All model-critical
packages retain the locked versions.

Run a one-trial-per-task smoke evaluation:

```bash
sbatch --array=0 --export=ALL,NUM_TRIALS=1,RUN_TAG=smoke \
  slurm/07_eval_openpi_reference.sbatch
```

Run the published four-suite evaluation:

```bash
sbatch --array=0-3 --export=ALL,NUM_TRIALS=50,RUN_TAG=official_d0_50 \
  slurm/07_eval_openpi_reference.sbatch
```

## Reproduction result

SLURM array job 41219255 completed 2,000 episodes successfully on H200 GPUs:

| Suite | Published | Reproduced | Successes |
| --- | ---: | ---: | ---: |
| Spatial | 98.8% | 98.4% | 492 / 500 |
| Object | 98.2% | 99.0% | 495 / 500 |
| Goal | 98.0% | 97.8% | 489 / 500 |
| LIBERO-10 | 92.4% | 93.4% | 467 / 500 |
| Macro average | 96.85% | 97.15% | 1,943 / 2,000 |

The reproduced macro average differs from the documented run by +0.30
percentage points, or six episode outcomes out of 2,000. The evaluator fixes
LIBERO initial states and NumPy seed but the policy is a stochastic
flow-matching sampler; the published table does not identify a fully pinned
software/hardware environment or a serialized inference-noise stream. The
reproduced score should therefore be treated as a successful statistical
reproduction, not a byte-for-byte replay of the unpublished rollout trace.

Wall times were 32:20 (Spatial), 33:36 (Object), 29:35 (Goal), and 58:16
(LIBERO-10). The per-task reproduced success rates were:

- Spatial: 100, 100, 100, 96, 94, 98, 100, 100, 100, 96%.
- Object: 96, 100, 98, 100, 100, 96, 100, 100, 100, 100%.
- Goal: 94, 100, 100, 94, 100, 98, 100, 100, 98, 94%.
- LIBERO-10: 94, 98, 98, 100, 100, 96, 98, 100, 56, 94%.

## OpenPI versus LeRobot conversion

The parity harness captures one fixed Spatial observation after the official
10 wait steps and supplies the same Gaussian noise to both models. It records
all ten Euler velocity fields and states, the final normalized action chunk,
and the first five environment actions.

Observation handling is equivalent to floating-point precision:

- Both flip the base and wrist cameras by 180 degrees, resize with padding to
  224 by 224, scale pixels to [-1, 1], and mask a third empty camera.
- Base and wrist image maximum absolute differences are 1.192e-7.
- Both camera masks are [true, true, false].

The saved LeRobot conversion is not semantically equivalent:

- Official pi05_libero uses 20 valid tokens for this task: cleaned task text,
  BOS, and a separately tokenized newline. With discrete_state_input=false,
  state is not embedded by the pi05 action expert.
- The saved LeRobot processor produces 154 valid tokens by inserting the
  normalized, zero-padded 32D state discretized into 256 bins in a
  Task/State/Action prompt.
- Official OpenPI predicts a 10-action chunk and executes the first five.
  The saved LeRobot checkpoint predicts 50 coupled action tokens and defaults
  to executing 10. The previous evaluator override and parity trace compare
  its first five actions against OpenPI.
- OpenPI always adds 1e-6 to each quantile range; LeRobot only substitutes
  1e-8 when a range is exactly zero. The resulting action unnormalization
  difference is negligible (maximum 9.537e-7).

When LeRobot is forced to the official 10-token horizon and official token
IDs, the final normalized action RMSE is 1.225e-3 and the first five executed
actions have RMSE 2.131e-3 (maximum 5.636e-3). With the saved 50-token,
state-in-prompt behavior, first-five action RMSE rises to 1.314e-1 (maximum
4.953e-1). Thus the conversion weights are numerically close; the material
regression comes from preprocessing and chunk semantics.

The separately instrumented OpenPI unrolled Euler graph differs from the
upstream compiled while-loop final by at most 0.001953125 in normalized space,
so intermediate velocity/state comparisons are diagnostic. All final
normalized and executed-action comparisons use the upstream compiled output.

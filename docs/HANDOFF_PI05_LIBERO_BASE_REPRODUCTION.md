# Handoff: reproduce pi05 LIBERO base performance first

## Mission

Before changing, fine-tuning, or evaluating CloudEdgeVLA, reproduce the published
LIBERO performance of the original Physical Intelligence OpenPI pi05 checkpoint
with the original OpenPI evaluator.

Do not fine-tune the base checkpoint. Do not use the LeRobot evaluator for this
first gate.

The Hugging Face checkpoint named `lerobot/pi05_libero_base` is a conversion.
The authoritative documented benchmark is produced by:

- OpenPI configuration: `pi05_libero`
- OpenPI checkpoint: `gs://openpi-assets/checkpoints/pi05_libero/`
- OpenPI LIBERO evaluator: `examples/libero/main.py`

Only after this reference gate succeeds should the LeRobot conversion be
evaluated or modified.

## Infrastructure

- Cluster login: `wxiong@login.rc.fas.harvard.edu`
- Project root: `/n/netscratch/ydu_lab/Lab/wxiong/EdgeCloud-VLA`
- Git repository: `git@github.com:XiongWenye/EdgeCloud-VLA.git`
- Working branch: `codex/pi05-cloudedge`
- Handoff baseline commit: `e99bed1`
- SLURM partition: `kempner_h200`
- SLURM account: `kempner_ydu_lab`
- GPU request: `--gres=gpu:nvidia_h200:1`

Do not run model, simulator, dataset, or evaluation work on the user's local
Mac. Cluster login-node work should be limited to lightweight inspection and
setup. Run inference and LIBERO simulation through SLURM.

## Existing pinned assets

The required assets are already present on cluster scratch.

### OpenPI source

Path:

```text
/n/netscratch/ydu_lab/Lab/wxiong/EdgeCloud-VLA/reference/openpi
```

Required revisions:

```text
OpenPI: 15a9616a00943ada6c20a0f158e3adb39df2ccac
LIBERO submodule: f78abd68ee283de9f9be3c8f7e2a9ad60246e95c
Aloha submodule: d1dc83afd89ded4379851257fe5d85632d31d5ec
```

Verify before running:

```bash
cd /n/netscratch/ydu_lab/Lab/wxiong/EdgeCloud-VLA
git -C reference/openpi rev-parse HEAD
git -C reference/openpi/third_party/libero rev-parse HEAD
```

Do not update the OpenPI checkout or its submodules during the reproduction.

### Official checkpoint

Local checkpoint path:

```text
/n/netscratch/ydu_lab/Lab/wxiong/EdgeCloud-VLA/cache/openpi_reference/openpi-assets/checkpoints/pi05_libero
```

The staged checkpoint contains 16 files totaling 12,439,085,481 bytes. The
policy must load normalization statistics from:

```text
assets/physical-intelligence/libero
```

inside that checkpoint directory.

### Tokenizer

```text
/n/netscratch/ydu_lab/Lab/wxiong/EdgeCloud-VLA/cache/openpi_reference/big_vision/paligemma_tokenizer.model
```

Expected SHA-256:

```text
8986bb4f423f07f8c7f70d0dbe3526fb2316056c17bae71b1ea975e77a168fc6
```

### Python environments

OpenPI model server:

```text
/n/netscratch/ydu_lab/Lab/wxiong/EdgeCloud-VLA/reference/openpi_server_env_pip/bin/python
```

Important model-stack versions:

```text
Python 3.11.15
JAX 0.5.3
Flax 0.10.2
Orbax 0.11.13
Torch 2.7.1+cu126
```

Official LIBERO client:

```text
/n/netscratch/ydu_lab/Lab/wxiong/EdgeCloud-VLA/reference/openpi/examples/libero/.venv/bin/python
```

The legacy client environment uses Python 3.8 and includes:

```text
numpy 1.22.4
torch 1.11.0+cu113
robosuite 1.4.1
libero 0.1.0
websockets 13.1
dm-tree 0.1.8
msgpack 1.1.0
```

Do not replace these environments unless a read-only verification shows they
are missing or corrupt.

## Authoritative evaluation protocol

Use exactly these settings:

| Setting | Value |
| --- | --- |
| Suites | `libero_spatial`, `libero_object`, `libero_goal`, `libero_10` |
| Tasks per suite | 10 |
| Initial states per task | 50 |
| Episodes per suite | 500 |
| Total episodes | 2,000 |
| NumPy/environment seed | 7 |
| Wait steps after reset | 10 |
| Model action horizon | 10 |
| Flow-matching Euler steps | 10 |
| Actions executed before replanning | 5 |
| Camera input | base plus wrist |
| Camera orientation | rotate both 180 degrees |
| Resize | resize with padding to 224 by 224 |
| Image dtype sent to server | uint8 |
| Spatial maximum steps | 220 |
| Object maximum steps | 280 |
| Goal maximum steps | 300 |
| LIBERO-10 maximum steps | 520 |

The model predicts ten actions but the evaluator executes only the first five
before requesting a new chunk.

## First gate: model-only smoke

The existing model smoke script verifies checkpoint restoration and a finite
`(10, 7)` action chunk:

```bash
cd /n/netscratch/ydu_lab/Lab/wxiong/EdgeCloud-VLA
sbatch \
  --job-name=openpi_model_smoke \
  --partition=kempner_h200 \
  --account=kempner_ydu_lab \
  --nodes=1 \
  --ntasks-per-node=1 \
  --cpus-per-task=8 \
  --gres=gpu:nvidia_h200:1 \
  --mem=96G \
  --time=01:00:00 \
  --output=logs/openpi_model_smoke_%j.out \
  --error=logs/openpi_model_smoke_%j.err \
  --wrap="cd /n/netscratch/ydu_lab/Lab/wxiong/EdgeCloud-VLA && \
    export OPENPI_DATA_HOME=/n/netscratch/ydu_lab/Lab/wxiong/EdgeCloud-VLA/cache/openpi_reference && \
    export PYTHONPATH=/n/netscratch/ydu_lab/Lab/wxiong/EdgeCloud-VLA/reference/openpi/src && \
    reference/openpi_server_env_pip/bin/python scripts/openpi_model_smoke.py \
      --checkpoint cache/openpi_reference/openpi-assets/checkpoints/pi05_libero"
```

Acceptance criterion:

```text
OPENPI_MODEL_SMOKE_OK (10, 7) True
```

A prior successful model smoke was job `41216405`.

## Second gate: end-to-end evaluator smoke

Run one initial state for each of the ten Spatial tasks:

```bash
cd /n/netscratch/ydu_lab/Lab/wxiong/EdgeCloud-VLA
sbatch \
  --array=0 \
  --export=ALL,NUM_TRIALS=1,RUN_TAG=handoff_smoke \
  slurm/07_eval_openpi_reference.sbatch
```

This is ten total episodes, not one total episode.

Acceptance criteria:

- SLURM state is `COMPLETED` with exit code `0:0`.
- Client log reaches ten completed episodes.
- Server log confirms the official checkpoint and norm-stat paths.
- The smoke should normally be near 10/10, but the mandatory condition is that
  all ten tasks run end to end without infrastructure errors.

Output locations:

```text
logs/openpi_ref_<jobid>_0.out
logs/openpi_ref_<jobid>_0.err
results/openpi_reference/handoff_smoke/libero_spatial/client.log
results/openpi_reference/handoff_smoke/libero_spatial/server.log
```

A prior successful end-to-end smoke was job `41218765`, which achieved 10/10.

## Third gate: full official reproduction

After both smoke gates pass:

```bash
cd /n/netscratch/ydu_lab/Lab/wxiong/EdgeCloud-VLA
sbatch \
  --array=0-3 \
  --export=ALL,NUM_TRIALS=50,RUN_TAG=handoff_official_d0_50 \
  slurm/07_eval_openpi_reference.sbatch
```

Array mapping:

| Array index | Suite |
| ---: | --- |
| 0 | `libero_spatial` |
| 1 | `libero_object` |
| 2 | `libero_goal` |
| 3 | `libero_10` |

Monitor without modifying jobs:

```bash
squeue -j <array_job_id> -o "%.18i %.12P %.10T %.12M %.10l %R"
sacct -j <array_job_id> \
  --format=JobID,JobName%24,State,Elapsed,ExitCode,MaxRSS
```

Extract final totals:

```bash
cd /n/netscratch/ydu_lab/Lab/wxiong/EdgeCloud-VLA
for i in 0 1 2 3; do
  f="logs/openpi_ref_<array_job_id>_${i}.out"
  echo "ARRAY ${i}"
  grep -a "Task suite:" "$f" | head -1
  grep -a "# episodes completed so far:" "$f" | tail -1
  grep -a "# successes:" "$f" | tail -1
  grep -a "Current total success rate:" "$f" | tail -1
done
```

Do not report partial progress as a final score. Each suite must show exactly
500 completed episodes.

## Expected scores

Published OpenPI documentation:

| Suite | Published success |
| --- | ---: |
| Spatial | 98.8% |
| Object | 98.2% |
| Goal | 98.0% |
| LIBERO-10 | 92.4% |
| Macro average | 96.85% |

Previous reproduction on this cluster, job `41219255`:

| Suite | Reproduced success | Successes |
| --- | ---: | ---: |
| Spatial | 98.4% | 492 / 500 |
| Object | 99.0% | 495 / 500 |
| Goal | 97.8% | 489 / 500 |
| LIBERO-10 | 93.4% | 467 / 500 |
| Macro average | 97.15% | 1,943 / 2,000 |

Prior wall times on H200 were:

| Suite | Wall time |
| --- | ---: |
| Spatial | 32:20 |
| Object | 33:36 |
| Goal | 29:35 |
| LIBERO-10 | 58:16 |

The exact 96.85% rollout trace is not published. The flow policy samples
Gaussian inference noise, and the published report does not serialize that
noise stream or fully pin the software/hardware environment. Therefore require
a close statistical reproduction, not byte-identical episode outcomes. Any
large shortfall, especially a result near the earlier LeRobot conversion
average of 92.5%, must be treated as a protocol or conversion mismatch.

## Known benign messages

These messages do not invalidate a successful run:

- Robosuite warning about a missing private `macros.py`.
- Gym deprecation warning.
- EGL errors from `EGLGLContext.__del__` after the evaluator has completed.
- A WebSocket `InvalidMessage` in the server log caused by the shell TCP
  readiness probe.
- JAX messages about unavailable ROCm or TPU backends on an NVIDIA node.

Judge success by the SLURM exit code and completed episode counts.

## Known failure modes and fixes

### LIBERO asks for a dataset directory

The job must export:

```text
LIBERO_CONFIG_PATH=/n/netscratch/ydu_lab/Lab/wxiong/EdgeCloud-VLA/config/openpi_reference/libero
```

The committed config file already points to the pinned LIBERO checkout.

### `ModuleNotFoundError: websockets.sync.client`

Use the existing client environment. It was repaired with
`websockets==13.1`, which supports Python 3.8.

### Server loads but client cannot connect

Check that both server and client run on the same allocated node and use the
port computed by `slurm/07_eval_openpi_reference.sbatch`. Do not expose the
server externally.

### Norm stats are missing

The server should log that it loaded norm stats from the checkpoint's
`assets/physical-intelligence/libero` directory. Stop if it silently evaluates
without those stats.

### Low success with LeRobot

Do not use that result to reject the official checkpoint. The saved LeRobot
conversion differs materially:

- It generates 154 valid tokens for the traced task versus OpenPI's 20.
- It inserts a discretized, padded 32D state into the prompt, while official
  `pi05_libero` uses task text plus newline and does not embed state.
- It predicts 50 coupled action tokens; official OpenPI predicts 10.
- The LeRobot artifact defaults to executing 10 actions; official OpenPI
  replans after five.

The conversion weights themselves are close when protocol inputs are aligned.
See `results/parity/openpi_vs_lerobot.md`.

## Reporting checklist

The handoff agent's report must include:

1. OpenPI commit and LIBERO submodule commit.
2. Exact checkpoint path.
3. SLURM job ID and final state for every array element.
4. Completed episodes and successes for every suite.
5. Per-suite and macro-average success rates.
6. Action horizon, flow steps, and replan interval.
7. Whether normalization stats loaded from the checkpoint.
8. Wall time per suite.
9. Any deviation from the committed evaluator script.
10. Direct links or cluster paths to client and server logs.

Do not call the base reproduction complete if any suite has fewer than 500
episodes or if a non-benign exception occurred before final totals were logged.

## Files to read

- `docs/openpi_reference_reproduction.md`: completed reproduction and parity findings.
- `slurm/07_eval_openpi_reference.sbatch`: authoritative evaluator job.
- `scripts/openpi_model_smoke.py`: checkpoint-only smoke.
- `scripts/capture_openpi_libero_observation.py`: canonical observation capture.
- `slurm/08_trace_openpi_lerobot_parity.sbatch`: optional conversion parity trace.
- `results/parity/openpi_vs_lerobot.md`: concise parity result.
- `results/parity/openpi_vs_lerobot.json`: machine-readable tensor metrics.

## Stop condition

Stop after the official OpenPI four-suite result is reproduced and reported.
Do not start base fine-tuning, CloudEdge training, delayed evaluation, or
Hugging Face upload unless the user explicitly authorizes the next phase.

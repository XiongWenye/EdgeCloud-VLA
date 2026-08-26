# Handoff: full CloudEdgeVLA reproduction on aligned pi05 LIBERO

## Directive to the receiving agent

Reproduce CloudEdgeVLA on the aligned pi05_libero_base, including the fresh
120k-step fine-tuning run and closed-loop LIBERO evaluation under simulated
observation delay. Work autonomously through implementation, validation, SLURM
submission, monitoring, aggregation, and documentation. Do not stop after
submitting jobs: inspect logs and artifacts, repair failures, and resume from
the latest complete checkpoint when possible.

The aligned base is already validated and is immutable. The older CloudEdge
bootstrap, checkpoint, and evaluation outputs were created before alignment and
are historical only. Do not use them as inputs or final results.

Do not run model inference, training, LIBERO simulation, or dataset processing
on the user's Mac. Use the Mac only as an SSH client. Lightweight downloads,
source inspection, environment setup, Git operations, and job submission may
run on the cluster login node. GPU work must run through SLURM.

Do not upload checkpoints/results to Hugging Face, publish a release, or delete
legacy artifacts unless the user separately authorizes it.

## Authoritative starting state

| Item | Required value |
| --- | --- |
| Cluster | wxiong@login.rc.fas.harvard.edu |
| Project | /n/netscratch/ydu_lab/Lab/wxiong/EdgeCloud-VLA |
| Git branch | codex/pi05-cloudedge |
| Starting commit | 7978e081d930bad1f6de4c4913f025468df6d968 |
| SLURM partition | kempner_h200 |
| SLURM account | kempner_ydu_lab |
| LeRobot commit | 8fff0fde7c79f23a93d845d1a50e985de01f8b8a |
| Dataset | lerobot/libero@a1aaacb7f6cd6ee5fb43120f673cebb0cfea7dd4 |
| Aligned base artifact | artifacts/pi05_libero_base_official_aligned |
| Edge encoder | google/siglip-base-patch16-224 |
| Edge revision | 7fd15f0689c79d79e38b1c2e2e2370a7bf2761ed |

Before changing anything:

~~~bash
cd /n/netscratch/ydu_lab/Lab/wxiong/EdgeCloud-VLA
git status --short --branch
git rev-parse HEAD
git submodule status third_party/lerobot
~~~

Preserve unrelated and untracked files. In particular, do not overwrite or
delete the existing results/baseline_h1_summary.* files.

## The aligned base gate is complete

| Suite | Aligned LeRobot | Official published |
| --- | ---: | ---: |
| Spatial | 98.60% (493/500) | 98.8% |
| Object | 99.00% (495/500) | 98.2% |
| Goal | 98.20% (491/500) | 98.0% |
| LIBERO-10 | 91.80% (459/500) | 92.4% |
| Macro average | **96.90%** | **96.85%** |

The cluster-side original OpenPI reference produced 97.15%. See:

- results/aligned_lerobot_summary.md
- docs/openpi_reference_reproduction.md
- docs/HANDOFF_PI05_LIBERO_BASE_REPRODUCTION.md

Do not fine-tune the base again. Do not revert to the raw Hugging Face
lerobot/pi05_libero_base conversion.

The aligned artifact contract is:

- predict 10 actions;
- execute 5 actions before replanning;
- use 10 flow-matching Euler steps;
- use OPENPI_QUANTILES for state and action and identity for images;
- use the official local PaliGemma tokenizer;
- do not insert state tokens into the language prompt;
- use two real cameras plus the official empty third-camera slot;
- preprocess at 224 by 224;
- use official episode limits, including 220 for Spatial.

## Invalid legacy lineage: preserve but never reuse

These paths predate alignment:

~~~text
artifacts/cloudedge_pi05_bootstrap_openpi_norm
outputs/cloudedge_pi05_libero_120k
outputs/invalid_cloudedge_pi05_libero_120k_lerobot_quantiles_job41108451
~~~

The old 120k checkpoint used a 50-action chunk, a different execution horizon,
the wrong quantile formula, and state-in-prompt preprocessing. Older
delay-labelled baseline folders also did not actually delay observations.
Never mix those metrics with the aligned run.

Use new lineage names:

~~~text
artifacts/cloudedge_pi05_bootstrap_official_aligned
outputs/cloudedge_pi05_libero_aligned_120k
results/baseline_aligned_delayed
results/cloudedge_aligned
~~~

## Reproduction contract

Complete the gates in order. A later gate may start only after the previous gate
passes. Commit and push implementation/configuration changes before long
training so every artifact has an exact source commit.

### Gate 1: deterministic setup

Update slurm/00_setup.sbatch. It currently checks out pinned LeRobot but does
not apply the committed alignment patch. Make setup idempotently:

1. check out LeRobot commit 8fff0fde7c79f23a93d845d1a50e985de01f8b8a;
2. apply patches/0001-align-lerobot-pi05-with-openpi-libero.patch if needed;
3. detect and accept an already-applied patch;
4. fail on unrelated dirty LeRobot state rather than silently resetting it;
5. run tests/test_pi05_alignment.py.

Record exact package versions and Git revisions in the log. Do not update
dependencies merely because newer versions exist.

### Gate 2: migrate CloudEdge preprocessing and bootstrap

Modify scripts/bootstrap_checkpoint.py and
src/lerobot_policy_cloudedge_pi05/processor_cloudedge_pi05.py so the new
bootstrap derives from artifacts/pi05_libero_base_official_aligned, not the raw
HF model.

The preprocessor must preserve aligned transforms and add only the temporal
operation needed by CloudEdge:

1. select the current state from the observation-history tensor;
2. apply aligned OpenPI state/action normalization;
3. use the local official PaliGemmaTokenizerProcessorStep;
4. do not use Pi05PrepareStateTokenizerProcessorStep;
5. package paligemma_tokenizer.model and stats into the bootstrap artifact.

Inherit chunk_size=10, n_action_steps=5, 10 flow steps, cameras, padding,
normalization, and prompt behavior from the aligned base. Add CloudEdge fields
without replacing inherited base fields with stale defaults.

Initialize the frozen cloud path from aligned base weights. Add the frozen
SigLIP edge encoder and a trainable, zero-initialized edge projection into the
action-expert embedding space.

Tests must cover:

- bootstrap config inheritance;
- tokenizer IDs and attention masks match aligned base;
- normalization/unnormalization match OpenPI quantiles;
- camera order and empty camera match aligned base;
- predicted chunk shape is (batch, 10, 7);
- execution length is 5;
- edge-off, d_max=0, fixed-noise output matches aligned base within a documented
  tolerance;
- zero-initialized edge projection leaves the bootstrap flow field unchanged.

Create a new bootstrap without overwriting the old artifact:

~~~text
artifacts/cloudedge_pi05_bootstrap_official_aligned
~~~

### Gate 3: validate delay semantics

Add deterministic tests or a synthetic trace proving:

- d_max=0 always selects the current cloud image;
- d_max>0 samples uniformly from integers 1 through d_max;
- delay is measured in environment steps, not policy calls;
- history updates every environment step, including while queued actions run;
- cloud receives image o[t-d];
- edge receives current image o[t];
- task text and proprioceptive state are current at t;
- warm-up uses the oldest valid frame, never an invalid/uninitialized frame.

For the delayed baseline, do not merely label result folders by delay. Use the
untrained aligned CloudEdge bootstrap with use_edge_vision=false. Its shared
history machinery supplies truly delayed cloud images while policy weights
remain the aligned base. At d_max=0, fixed-noise output must match aligned base.

### Gate 4: H200 smoke gates

Run through SLURM on one H200:

1. bootstrap load and one finite inference chunk;
2. at least two optimizer steps, checkpoint save/load, then one resumed step;
3. one episode from each of ten Spatial tasks at d_max=0;
4. a d_max=10 trace logging sampled delays and selected history indices.

Acceptance:

- SLURM COMPLETED with exit 0:0;
- no NaN/Inf loss or actions;
- chunk shape and execution length match the aligned contract;
- logs prove aligned artifact and local tokenizer were loaded;
- delay trace proves observations change as intended;
- smoke checkpoint resumes successfully.

### Gate 5: aligned 120k fine-tuning

| Parameter | Value |
| --- | --- |
| Initial weights | aligned pi05 CloudEdge bootstrap |
| Steps | 120,000 optimizer steps |
| History window | 21 frames |
| Training stale delay | Uniform integers 1 through 20 |
| Fresh/stale target | same current 10-step action chunk |
| Flow noise/time | identical for fresh and stale branches |
| Stale-loss maximum coefficient | 0.5 |
| Stale curriculum | linear 0 to 0.5 over 10,000 steps |
| Global batch | 8: 2 per GPU times 4 GPUs |
| Seed | 7 |
| Precision | bfloat16 |
| Learning rate | 2.5e-5 |
| LR warm-up | 1,000 steps |
| LR decay horizon/floor | 30,000 steps / 2.5e-6 |
| Save interval | 5,000 steps |
| Cloud vision encoder | frozen |
| Edge vision encoder | frozen |
| train_expert_only | false |
| Gradient checkpointing | false unless a documented memory issue requires it |
| Compile | false |
| Data workers | 4 |
| GPUs | 4 H200 |
| Output | outputs/cloudedge_pi05_libero_aligned_120k |

The paper does not disclose lambda_max, curriculum duration, optimizer, learning
rate, batch size, precise execution horizon, or exact checkpoint revisions.
The values above are the repository's declared pi05 reproduction choices, not
claims about hidden paper settings. Preserve them for comparability with the
existing implementation and record any necessary deviation before running it.

Preserve paired flow-matching supervision. For action a, noise epsilon, and time
tau, both branches use the same x_tau = tau*epsilon + (1-tau)*a and target
u_tau = epsilon-a. Never independently resample noise or time across branches.

Use a new SLURM path and request kempner_h200, account kempner_ydu_lab, one
node, one task, and four H200 GPUs. The old run took about 11.5 hours; request
enough wall time for validation and final checkpoint writing.

Write a run manifest with source commit, base artifact, dataset revision,
parameters, package versions, job ID, and timestamps. Monitor each save
interval. On failure, diagnose logs and resume only from the newest complete
checkpoint.

### Gate 6: mandatory closed-loop matrix

Evaluate:

~~~text
Suites: libero_spatial, libero_object, libero_goal, libero_10
d_max: 0, 10, 20, 40
CloudEdge seeds: 7, 8, 9
Delayed aligned-baseline seed: 7
Episodes: 50 initial states per task, 10 tasks per suite
~~~

Each suite evaluation is 500 episodes. The mandatory core is:

- delayed baseline: 4 delays x 4 suites x 1 seed = 16 records;
- trained CloudEdge: 4 delays x 4 suites x 3 seeds = 48 records;
- total: 64 complete records.

Retain predict-10/execute-5, 10 flow steps, official tokenizer, normalization,
cameras, and episode limits. Use H200 arrays and unique paths. Reduce array
concurrency if LIBERO or the filesystem is unstable.

Recommended layout:

~~~text
results/baseline_aligned_delayed/raw/dmax_<D>/<suite>/seed_7/eval_info.json
results/cloudedge_aligned/raw/dmax_<D>/<suite>/seed_<S>/eval_info.json
~~~

Every record must include source commit, checkpoint and step, method, suite,
seed, d_max, episode/success counts, success rate, chunk size, execution
horizon, flow steps, sampled-delay rule, job ID, and completion timestamp.

Before accepting the matrix, verify the d_max=0 baseline remains statistically
consistent with the established 96.90%. A material regression is an evaluator
or preprocessing bug, not a CloudEdge finding.

### Gate 7: complete delay curve

Run additional d_max values 5, 15, 25, and 30 with the same suites, episodes,
baseline seed, and three CloudEdge seeds. Combined with Gate 6, the curve is:

~~~text
0, 5, 10, 15, 20, 25, 30, 40
~~~

### Gate 8: mechanism diagnostic

On LIBERO Spatial:

- use one demonstration from each of 10 tasks;
- select eight states per demonstration, 80 states total;
- evaluate fixed delays 0, 1, 3, 5, 8, 10, 15, and 20;
- keep proprioception and text current;
- compare fresh/stale-cloud predictions with current edge vision;
- use identical flow noise/time for comparisons;
- save state IDs, seeds/tensors, raw actions, and aggregate metrics.

The paper reports normalized 8-step, 7-D chunks; aligned pi05 predicts 10 steps.
Do not silently truncate. Report the native 10-step diagnostic and optionally a
clearly labelled first-eight-step view derived from the same saved predictions.

### Gate 9: trained ablations at d_max=10

Reproduce:

1. no edge vision;
2. frozen SigLIP-Base edge encoder, the main setting;
3. frozen SigLIP-SO400M edge encoder;
4. fresh-only loss;
5. stale-only loss;
6. joint fresh/stale loss, the main setting.

Every trained ablation starts from the aligned base and has distinct config,
output, and manifest. Use lambda=0 for fresh-only; stale-only must contain no
hidden fresh term; joint uses the main schedule. Do not label an inference-only
toggle as a trained ablation.

These are expensive 120k runs. Submit them after the main pipeline is proven.
Evaluate at d_max=10 using the same LIBERO protocol and seeds. Report compute
and deviations.

### Gate 10: aggregate and report

Update or replace scripts/aggregate_results.py to validate the expected grid
before calculating headlines. Fail on missing, duplicate, truncated, or
mismatched records.

Report:

- per-suite and macro success at every delay;
- CloudEdge mean and sample standard deviation across seeds 7, 8, 9;
- delayed baseline seed-7 values;
- normalized trapezoidal delay AUC from 0 through 40;
- retention at 40 = 100 * success(40) / success(0);
- successes and episode counts behind every percentage;
- comparison with the aligned no-delay gate;
- training curves, wall time, GPU-hours, checkpoint identity;
- failed/retried jobs and protocol deviations.

Produce machine-readable CSV/JSON and a Markdown report. Separate newly measured
pi05 CloudEdge results, the 96.90% pi05 base gate, and paper OpenVLA-OFT
references.

Paper values are references, not pi05 targets: at d_max=40, Spatial 76.4%,
Object 75.6%, Goal 78.0%, Long 63.8%; macro delay AUC 90.8%; retention 76.5%.
Never present these as measured pi05 results.

## Repository fixes required before submission

Update configs/experiment.json and README.md. They contain stale pre-alignment
settings. They must name the aligned base, predict-10/execute-5, OpenPI
quantiles, local tokenizer, new paths, and actual checkpointing choice.

Update scripts/static_validate.py so it checks the aligned manifest instead of
requiring n_action_steps=1. Validate all 64 core records and later the full
curve.

Prefer new aligned SLURM filenames over mutating legacy scripts whose job IDs
appear in reports. If a name is reused, document its boundary commit.

## Dependency rule

Preserve this order:

~~~text
setup -> bootstrap -> smoke -> 120k training -> core evaluation
      -> full delay curve -> diagnostics/ablations -> aggregation
~~~

Use SLURM afterok dependencies, never afterany, between scientific stages. A
failed prerequisite must block downstream jobs.

## Completion checklist

- [ ] Fresh setup applies/verifies the alignment patch and passes tests.
- [ ] Bootstrap derives only from aligned base.
- [ ] Fixed-noise d=0 edge-off output matches aligned base.
- [ ] Delay tests prove environment-step semantics.
- [ ] H200 train/eval smokes pass.
- [ ] Main aligned 120k checkpoint and manifest complete.
- [ ] All 64 mandatory evaluation records validate.
- [ ] Full eight-point delay curve completes.
- [ ] Mechanism diagnostic saves raw predictions.
- [ ] Requested trained ablations and d=10 evaluations complete.
- [ ] Aggregated CSV/JSON and Markdown report are committed.
- [ ] README/configs match the actual run.
- [ ] Commits are pushed to codex/pi05-cloudedge.
- [ ] Legacy outputs remain preserved/excluded.
- [ ] Real-robot results are marked unavailable, not fabricated.

## Status format

Every user update must give evidence:

~~~text
Phase:
Git commit:
Submitted/running/completed job IDs:
Latest checkpoint step:
Observed metric or failure:
Artifact/result paths:
Next automatic action:
~~~

If jobs are running, report elapsed time, step/episode progress, and a defensible
ETA. If waiting will exceed one hour, tell the user it is safe to pause the
Codex task and return after jobs finish.

## Scope boundary

The cluster work covers simulation, fine-tuning, diagnostics, and ablations.
The Franka real-robot pilot cannot be reproduced without hardware, task assets,
calibration, and authorization. Mark it out of scope unless the user supplies
those resources. Never infer or fabricate real-robot success rates.

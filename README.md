# CloudEdge π0.5 on LIBERO

This repository ports the paired-frame CloudEdgeVLA method to the LeRobot π0.5 flow-matching policy. It is an experimental reproduction, not the authors' OpenVLA-OFT implementation.

The adaptation is made at the conditional flow field. A cloud π0.5 prefix encodes a current or delayed image and language into per-layer KV context. A frozen SigLIP-Base encoder processes the current edge image, and a trainable zero-initialized projection adds that current visual condition to every noisy action token in the π0.5 action expert. Fresh and stale paths use the same action chunk, Gaussian noise, and sampled flow time:

`L = (1 - λ) MSE(v_fresh, u_t) + λ MSE(v_stale, u_t)`.

This preserves the original π0.5 denoising objective while implementing the temporal asymmetry of CloudEdgeVLA. Current proprioception is never delayed.

## Reproduction contract

- LeRobot: `v0.4.4`, commit `8fff0fde7c79f23a93d845d1a50e985de01f8b8a`
- Base: `lerobot/pi05_libero_base@a217bfd3b14673cf2ce597e69997ab21866438dd`
- Dataset: `lerobot/libero@a1aaacb7f6cd6ee5fb43120f673cebb0cfea7dd4`
- Edge encoder: `google/siglip-base-patch16-224@7fd15f0689c79d79e38b1c2e2e2370a7bf2761ed`
- Training window: 21 frames, with stale offset sampled uniformly from 1 through 20
- Evaluation: `d_max ∈ {0,10,20,40}`, 50 trials/task, suites Spatial/Object/Goal/Long
- CloudEdge seeds: 7, 8, 9; π0.5 delayed single-path baseline: seed 7
- Replan after every action (`n_action_steps=1`)

The uploaded paper does not disclose λmax, curriculum length, optimizer/batch details, or its exact pretrained checkpoint revision. The declared defaults are λmax 0.5, 10k warmup steps, 120k optimizer steps, and global batch 8. See [configs/experiment.json](configs/experiment.json) and [paper_protocol.md](paper_protocol.md).

## Cluster workflow

From `/n/netscratch/ydu_lab/Lab/wxiong/EdgeCloud-VLA`:

```bash
setup_job=$(sbatch --parsable slurm/00_setup.sbatch)
bootstrap_job=$(sbatch --parsable --dependency=afterok:$setup_job slurm/01_bootstrap.sbatch)
smoke_job=$(sbatch --parsable --dependency=afterok:$bootstrap_job slurm/02_smoke.sbatch)
eval_smoke_job=$(sbatch --parsable --dependency=afterok:$smoke_job slurm/02_eval_smoke.sbatch)
train_job=$(sbatch --parsable --dependency=afterok:$eval_smoke_job slurm/03_train.sbatch)
baseline_job=$(sbatch --parsable --dependency=afterok:$eval_smoke_job slurm/05_eval_pi05_baseline.sbatch)
cloudedge_job=$(sbatch --parsable --dependency=afterok:$train_job slurm/04_eval_cloudedge.sbatch)
aggregate_job=$(sbatch --parsable \
  --dependency=afterok:${baseline_job}_\*:${cloudedge_job}_\* \
  slurm/06_aggregate.sbatch)
upload_job=$(sbatch --parsable --dependency=afterok:$aggregate_job slurm/06_upload.sbatch)
```

To re-aggregate manually after both evaluation arrays finish:

```bash
source .venv/bin/activate
python scripts/aggregate_results.py results/raw \
  --output results/summary.csv \
  --markdown-output results/summary.md \
  --validate-paper-grid
```

The 120k job checkpoints every 5k updates and requeues five minutes before the 12-hour wall time. It resumes from `checkpoints/last`.

## What is and is not reproduced

The code reproduces paired episode-safe frames, fresh/stale dual supervision, current edge vision, nonblocking cloud-context/edge-action APIs, uniform-delay closed-loop evaluation, and the paper's suite/trial/seed grid. It does not claim that π0.5 numbers should equal the paper's OpenVLA-OFT numbers. The changed backbone, flow objective, 50-step π0.5 action horizon, and disclosed assumptions make this a new experimental result.

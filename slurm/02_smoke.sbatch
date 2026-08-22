#!/bin/bash
#SBATCH --job-name=cepi05_smoke
#SBATCH --output=/n/netscratch/ydu_lab/Lab/wxiong/EdgeCloud-VLA/logs/smoke_%j.out
#SBATCH --error=/n/netscratch/ydu_lab/Lab/wxiong/EdgeCloud-VLA/logs/smoke_%j.err
#SBATCH --partition=kempner
#SBATCH --account=kempner_ydu_lab
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH --time=02:00:00

set -euo pipefail
export PROJECT_ROOT=/n/netscratch/ydu_lab/Lab/wxiong/EdgeCloud-VLA
export HF_HOME="$PROJECT_ROOT/cache/huggingface"
export HF_LEROBOT_HOME="$PROJECT_ROOT/cache/lerobot"
export MUJOCO_GL=egl
export WANDB_MODE=disabled
export PYTHONNOUSERSITE=1
source "$PROJECT_ROOT/.venv/bin/activate"

OUT="$PROJECT_ROOT/outputs/smoke_${SLURM_JOB_ID}"
lerobot-train \
  --policy.path="$PROJECT_ROOT/artifacts/cloudedge_pi05_bootstrap" \
  --policy.device=cuda \
  --policy.dtype=bfloat16 \
  --policy.gradient_checkpointing=true \
  --policy.compile_model=false \
  --policy.push_to_hub=false \
  --dataset.repo_id=lerobot/libero \
  --dataset.revision=a1aaacb7f6cd6ee5fb43120f673cebb0cfea7dd4 \
  --dataset.video_backend=pyav \
  --steps=2 \
  --batch_size=1 \
  --num_workers=2 \
  --save_checkpoint=true \
  --save_freq=2 \
  --eval_freq=0 \
  --wandb.enable=false \
  --output_dir="$OUT" \
  --job_name=cloudedge_pi05_smoke

test -s "$OUT/checkpoints/000002/pretrained_model/model.safetensors"
ln -sfn "$OUT/checkpoints/000002/pretrained_model" "$PROJECT_ROOT/artifacts/cloudedge_pi05_smoke"
echo "SMOKE_SUCCESS=true"

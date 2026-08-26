#!/bin/bash
#SBATCH --job-name=cepi05_h200_smoke
#SBATCH --output=/n/netscratch/ydu_lab/Lab/wxiong/EdgeCloud-VLA/logs/h200_smoke_%j.out
#SBATCH --error=/n/netscratch/ydu_lab/Lab/wxiong/EdgeCloud-VLA/logs/h200_smoke_%j.err
#SBATCH --partition=kempner_h200
#SBATCH --account=kempner_ydu_lab
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:nvidia_h200:4
#SBATCH --mem=192G
#SBATCH --time=00:30:00

set -euo pipefail
export PROJECT_ROOT=/n/netscratch/ydu_lab/Lab/wxiong/EdgeCloud-VLA
export HF_HOME="$PROJECT_ROOT/cache/huggingface"
export HF_LEROBOT_HOME="$PROJECT_ROOT/cache/lerobot"
export WANDB_MODE=disabled
export PYTHONNOUSERSITE=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
source "$PROJECT_ROOT/.venv/bin/activate"

nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
OUT="$PROJECT_ROOT/outputs/h200_smoke_${SLURM_JOB_ID}"
accelerate launch --num_processes=4 -m lerobot.scripts.lerobot_train \
  --policy.path="$PROJECT_ROOT/artifacts/cloudedge_pi05_bootstrap" \
  --policy.device=cuda \
  --policy.dtype=bfloat16 \
  --policy.freeze_vision_encoder=true \
  --policy.train_expert_only=false \
  --policy.gradient_checkpointing=false \
  --policy.compile_model=false \
  --policy.push_to_hub=false \
  --dataset.repo_id=lerobot/libero \
  --dataset.revision=a1aaacb7f6cd6ee5fb43120f673cebb0cfea7dd4 \
  --dataset.video_backend=pyav \
  --steps=10 \
  --batch_size=2 \
  --num_workers=4 \
  --save_checkpoint=false \
  --eval_freq=0 \
  --wandb.enable=false \
  --output_dir="$OUT" \
  --job_name=cloudedge_pi05_h200_smoke

echo "H200_SMOKE_SUCCESS=true"

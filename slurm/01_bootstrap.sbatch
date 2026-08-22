#!/bin/bash
#SBATCH --job-name=cepi05_bootstrap
#SBATCH --output=/n/netscratch/ydu_lab/Lab/wxiong/EdgeCloud-VLA/logs/bootstrap_%j.out
#SBATCH --error=/n/netscratch/ydu_lab/Lab/wxiong/EdgeCloud-VLA/logs/bootstrap_%j.err
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
export PYTHONNOUSERSITE=1
source "$PROJECT_ROOT/.venv/bin/activate"

python "$PROJECT_ROOT/scripts/bootstrap_checkpoint.py" \
  --base=lerobot/pi05_libero_base \
  --base-revision=a217bfd3b14673cf2ce597e69997ab21866438dd \
  --dataset=lerobot/libero \
  --dataset-revision=a1aaacb7f6cd6ee5fb43120f673cebb0cfea7dd4 \
  --output="$PROJECT_ROOT/artifacts/cloudedge_pi05_bootstrap" \
  --device=cuda \
  --history-window=21 \
  --lambda-max=0.5 \
  --warmup-steps=10000


#!/bin/bash
#SBATCH --job-name=cepi05_bootstrap
#SBATCH --output=/n/netscratch/ydu_lab/Lab/wxiong/EdgeCloud-VLA/logs/bootstrap_%j.out
#SBATCH --error=/n/netscratch/ydu_lab/Lab/wxiong/EdgeCloud-VLA/logs/bootstrap_%j.err
#SBATCH --partition=kempner_h200
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

NORM_STATS="$PROJECT_ROOT/cache/openpi_reference/openpi-assets/checkpoints/pi05_libero/assets/physical-intelligence/libero/norm_stats.json"
BASE_CHECKPOINT="$PROJECT_ROOT/artifacts/pi05_libero_base_official_aligned"
BOOTSTRAP_OUTPUT="$PROJECT_ROOT/artifacts/cloudedge_pi05_bootstrap_official_aligned"

mkdir -p "$PROJECT_ROOT/artifacts" "$PROJECT_ROOT/logs"

nvidia-smi --query-gpu=index,name,uuid,memory.total --format=csv,noheader
python - <<PY
import torch

assert torch.cuda.is_available(), "CUDA is not available inside the allocated SLURM job"
print(f"cuda_device_count={torch.cuda.device_count()}")
print(f"cuda_device_name={torch.cuda.get_device_name(0)}")
torch.ones(1, device="cuda")
print("CUDA_PREFLIGHT_SUCCESS=true")
PY

python "$PROJECT_ROOT/scripts/bootstrap_checkpoint.py" \
  --base="$BASE_CHECKPOINT" \
  --norm-stats="$NORM_STATS" \
  --output="$BOOTSTRAP_OUTPUT" \
  --device=cuda \
  --history-window=21 \
  --lambda-max=0.5 \
  --warmup-steps=10000

echo "BOOTSTRAP_JOB_COMPLETE=true"

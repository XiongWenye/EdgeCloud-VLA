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

NORM_STATS="$PROJECT_ROOT/artifacts/openpi_pi05_libero_norm_stats.json"
mkdir -p "$PROJECT_ROOT/artifacts"
curl -L --fail --retry 5 --silent --show-error \
  -o "$NORM_STATS" \
  "https://storage.googleapis.com/openpi-assets/checkpoints/pi05_libero/assets/physical-intelligence/libero/norm_stats.json"
echo "b3a44bb2810436fb62917decaea58bd4d9110255df527dea21e8fd40c960bd84  $NORM_STATS" \
  | sha256sum --check --strict

nvidia-smi --query-gpu=index,name,uuid,memory.total --format=csv,noheader
python - <<'PY'
import torch

assert torch.cuda.is_available(), "CUDA is not available inside the allocated SLURM job"
print(f"cuda_device_count={torch.cuda.device_count()}")
print(f"cuda_device_name={torch.cuda.get_device_name(0)}")
torch.ones(1, device="cuda")
print("CUDA_PREFLIGHT_SUCCESS=true")
PY

python "$PROJECT_ROOT/scripts/bootstrap_checkpoint.py" \
  --base=lerobot/pi05_libero_base \
  --base-revision=a217bfd3b14673cf2ce597e69997ab21866438dd \
  --norm-stats="$NORM_STATS" \
  --output="$PROJECT_ROOT/artifacts/cloudedge_pi05_bootstrap_openpi_norm" \
  --device=cuda \
  --history-window=21 \
  --lambda-max=0.5 \
  --warmup-steps=10000

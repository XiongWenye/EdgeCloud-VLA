#!/bin/bash
#SBATCH --job-name=cepi05_v3_boot
#SBATCH --output=/n/netscratch/ydu_lab/Lab/wxiong/EdgeCloud-VLA/logs/v3_bootstrap_%j.out
#SBATCH --error=/n/netscratch/ydu_lab/Lab/wxiong/EdgeCloud-VLA/logs/v3_bootstrap_%j.err
#SBATCH --partition=kempner_h200
#SBATCH --account=kempner_ydu_lab
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:nvidia_h200:1
#SBATCH --mem=96G
#SBATCH --time=02:00:00

set -euo pipefail
PROJECT_ROOT=/n/netscratch/ydu_lab/Lab/wxiong/EdgeCloud-VLA
CODE_ROOT=${CODE_ROOT:-$PROJECT_ROOT}
export HF_HOME=$PROJECT_ROOT/cache/huggingface
export HF_LEROBOT_HOME=$PROJECT_ROOT/cache/lerobot
export PYTHONNOUSERSITE=1
export PYTHONPATH=$CODE_ROOT/src:${PYTHONPATH:-}
source $PROJECT_ROOT/.venv/bin/activate
cd $CODE_ROOT

OUTPUT=$PROJECT_ROOT/artifacts/cloudedge_pi05_v3_bootstrap_official_aligned
test ! -e $OUTPUT
python $CODE_ROOT/scripts/bootstrap_checkpoint_v3.py \
  --base=$PROJECT_ROOT/artifacts/pi05_libero_base_official_aligned \
  --norm-stats=$PROJECT_ROOT/cache/openpi_reference/openpi-assets/checkpoints/pi05_libero/assets/physical-intelligence/libero/norm_stats.json \
  --output=$OUTPUT \
  --device=cuda

test -s $OUTPUT/model.safetensors
echo CLOUDEDGE_V3_BOOTSTRAP_SUCCESS=true

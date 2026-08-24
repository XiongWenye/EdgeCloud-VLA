#!/bin/bash
#SBATCH --job-name=cepi05_v2_smoke
#SBATCH --output=/n/netscratch/ydu_lab/Lab/wxiong/EdgeCloud-VLA/logs/v2_smoke_%j.out
#SBATCH --error=/n/netscratch/ydu_lab/Lab/wxiong/EdgeCloud-VLA/logs/v2_smoke_%j.err
#SBATCH --partition=kempner_h200
#SBATCH --account=kempner_ydu_lab
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:nvidia_h200:1
#SBATCH --mem=128G
#SBATCH --time=02:00:00

set -euo pipefail
PROJECT_ROOT=/n/netscratch/ydu_lab/Lab/wxiong/EdgeCloud-VLA
CODE_ROOT=${CODE_ROOT:-$PROJECT_ROOT}
export HF_HOME=$PROJECT_ROOT/cache/huggingface
export HF_LEROBOT_HOME=$PROJECT_ROOT/cache/lerobot
export LIBERO_CONFIG_PATH=$PROJECT_ROOT/configs/libero
export MUJOCO_GL=egl
export PYTHONNOUSERSITE=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH=$CODE_ROOT/src:${PYTHONPATH:-}
source $PROJECT_ROOT/.venv/bin/activate
cd $CODE_ROOT

OUTPUT=$PROJECT_ROOT/outputs/cloudedge_pi05_v2_smoke_$SLURM_JOB_ID
python $CODE_ROOT/scripts/smoke_cloudedge_v2_h200.py \
  --bootstrap=$PROJECT_ROOT/artifacts/cloudedge_pi05_v2_bootstrap_official_aligned \
  --aligned-base=$PROJECT_ROOT/artifacts/pi05_libero_base_official_aligned \
  --output=$OUTPUT

test -s $OUTPUT/report.json
echo CLOUDEDGE_V2_SMOKE_JOB_SUCCESS=true

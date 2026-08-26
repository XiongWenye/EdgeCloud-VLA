#!/bin/bash
#SBATCH --job-name=pi05_aligned_smoke
#SBATCH --output=/n/netscratch/ydu_lab/Lab/wxiong/EdgeCloud-VLA/logs/eval_aligned_smoke_%j.out
#SBATCH --error=/n/netscratch/ydu_lab/Lab/wxiong/EdgeCloud-VLA/logs/eval_aligned_smoke_%j.err
#SBATCH --partition=kempner_h200
#SBATCH --account=kempner_ydu_lab
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:nvidia_h200:1
#SBATCH --mem=96G
#SBATCH --time=01:00:00

set -euo pipefail
export PROJECT_ROOT=/n/netscratch/ydu_lab/Lab/wxiong/EdgeCloud-VLA
export HF_HOME="${PROJECT_ROOT}/cache/huggingface"
export HF_LEROBOT_HOME="${PROJECT_ROOT}/cache/lerobot"
export LIBERO_CONFIG_PATH="${PROJECT_ROOT}/configs/libero"
export MUJOCO_GL=egl
export PYTHONNOUSERSITE=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
source "${PROJECT_ROOT}/.venv/bin/activate"

CHECKPOINT="${PROJECT_ROOT}/artifacts/pi05_libero_base_official_aligned"
OUT="${PROJECT_ROOT}/results/smoke_aligned_spatial"
mkdir -p "${OUT}" "${PROJECT_ROOT}/logs"

lerobot-eval   --policy.path="${CHECKPOINT}"   --policy.device=cuda   --env.type=libero   --env.task=libero_spatial   --env.control_mode=relative   --env.init_states=true   --env.max_parallel_tasks=1   --eval.batch_size=1   --eval.n_episodes=1   --seed=7   --output_dir="${OUT}"

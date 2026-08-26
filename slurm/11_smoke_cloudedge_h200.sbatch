#!/bin/bash
#SBATCH --job-name=cepi05_smoke_g4
#SBATCH --output=/n/netscratch/ydu_lab/Lab/wxiong/EdgeCloud-VLA/logs/smoke_g4_%j.out
#SBATCH --error=/n/netscratch/ydu_lab/Lab/wxiong/EdgeCloud-VLA/logs/smoke_g4_%j.err
#SBATCH --partition=kempner_h200
#SBATCH --account=kempner_ydu_lab
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH --time=01:00:00

set -euo pipefail
export PROJECT_ROOT=/n/netscratch/ydu_lab/Lab/wxiong/EdgeCloud-VLA
export HF_HOME="$PROJECT_ROOT/cache/huggingface"
export HF_LEROBOT_HOME="$PROJECT_ROOT/cache/lerobot"
export LIBERO_CONFIG_PATH="$PROJECT_ROOT/configs/libero"
export MUJOCO_GL=egl
export PYTHONNOUSERSITE=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
source "$PROJECT_ROOT/.venv/bin/activate"

BOOTSTRAP="$PROJECT_ROOT/artifacts/cloudedge_pi05_bootstrap_official_aligned"
OUT_SMOKE="$PROJECT_ROOT/outputs/h200_smoke_gate4"
OUT_EVAL="$PROJECT_ROOT/results/smoke_aligned_spatial_ce"
mkdir -p "$OUT_SMOKE" "$OUT_EVAL" "$PROJECT_ROOT/logs"

# 1. Run Python smoke test (inference chunk, 2 optimizer steps, checkpoint save/load/resume, delay trace)
python "$PROJECT_ROOT/scripts/smoke_cloudedge_h200.py" \
  --bootstrap="$BOOTSTRAP" \
  --aligned-base="$PROJECT_ROOT/artifacts/pi05_libero_base_official_aligned" \
  --smoke-dir="$OUT_SMOKE"

# 2. Run 1 episode rollout on each of 10 tasks in LIBERO Spatial at d_max=0
lerobot-eval \
  --policy.path="$BOOTSTRAP" \
  --policy.device=cuda \
  --policy.eval_delay_max=0 \
  --env.type=libero \
  --env.task=libero_spatial \
  --env.control_mode=relative \
  --env.init_states=true \
  --env.max_parallel_tasks=1 \
  --eval.batch_size=1 \
  --eval.n_episodes=1 \
  --seed=7 \
  --output_dir="$OUT_EVAL"

echo "GATE4_H200_SMOKE_SUCCESS=true"

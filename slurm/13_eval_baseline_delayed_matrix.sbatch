#!/bin/bash
#SBATCH --job-name=eval_base_mat
#SBATCH --output=/n/netscratch/ydu_lab/Lab/wxiong/EdgeCloud-VLA/logs/eval_base_mat_%A_%a.out
#SBATCH --error=/n/netscratch/ydu_lab/Lab/wxiong/EdgeCloud-VLA/logs/eval_base_mat_%A_%a.err
#SBATCH --partition=kempner_h200
#SBATCH --account=kempner_ydu_lab
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --array=0-15

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

# Baseline matrix parameters: 4 suites x 4 delays x 1 seed (7) = 16 tasks
SUITES=(libero_spatial libero_object libero_goal libero_10)
DELAYS=(0 10 20 40)

TOTAL_DELAYS=${#DELAYS[@]}

delay_idx=$(( SLURM_ARRAY_TASK_ID % TOTAL_DELAYS ))
suite_idx=$(( SLURM_ARRAY_TASK_ID / TOTAL_DELAYS ))

suite=${SUITES[$suite_idx]}
d_max=${DELAYS[$delay_idx]}
seed=7

BOOTSTRAP="$PROJECT_ROOT/artifacts/cloudedge_pi05_bootstrap_official_aligned"
OUT="$PROJECT_ROOT/results/baseline_aligned_delayed/raw/dmax_${d_max}/${suite}/seed_${seed}"
mkdir -p "$OUT" "$PROJECT_ROOT/logs"

echo "Evaluating Delayed Aligned Baseline: suite=$suite, d_max=$d_max, seed=$seed"
echo "Policy: $BOOTSTRAP (use_edge_vision=false)"
echo "Output: $OUT"

lerobot-eval \
  --policy.path="$BOOTSTRAP" \
  --policy.device=cuda \
  --policy.eval_delay_max="$d_max" \
  --policy.use_edge_vision=false \
  --env.type=libero \
  --env.task="$suite" \
  --env.control_mode=relative \
  --env.init_states=true \
  --env.max_parallel_tasks=1 \
  --eval.batch_size=10 \
  --eval.n_episodes=50 \
  --seed="$seed" \
  --output_dir="$OUT"

echo "BASELINE_EVAL_COMPLETE: suite=$suite, d_max=$d_max, seed=$seed"

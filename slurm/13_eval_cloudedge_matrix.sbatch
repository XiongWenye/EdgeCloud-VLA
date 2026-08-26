#!/bin/bash
#SBATCH --job-name=eval_ce_mat
#SBATCH --output=/n/netscratch/ydu_lab/Lab/wxiong/EdgeCloud-VLA/logs/eval_ce_mat_%A_%a.out
#SBATCH --error=/n/netscratch/ydu_lab/Lab/wxiong/EdgeCloud-VLA/logs/eval_ce_mat_%A_%a.err
#SBATCH --partition=kempner_h200
#SBATCH --account=kempner_ydu_lab
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --array=0-47

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

# Matrix parameters: 4 suites x 4 delays x 3 seeds = 48 tasks
SUITES=(libero_spatial libero_object libero_goal libero_10)
DELAYS=(0 10 20 40)
SEEDS=(7 8 9)

TOTAL_SEEDS=${#SEEDS[@]}
TOTAL_DELAYS=${#DELAYS[@]}

# Compute indices from SLURM_ARRAY_TASK_ID
seed_idx=$(( SLURM_ARRAY_TASK_ID % TOTAL_SEEDS ))
rem=$(( SLURM_ARRAY_TASK_ID / TOTAL_SEEDS ))
delay_idx=$(( rem % TOTAL_DELAYS ))
suite_idx=$(( rem / TOTAL_DELAYS ))

suite=${SUITES[$suite_idx]}
d_max=${DELAYS[$delay_idx]}
seed=${SEEDS[$seed_idx]}

CHECKPOINT="${1:-$PROJECT_ROOT/outputs/cloudedge_pi05_libero_aligned_120k/checkpoints/120000/pretrained_model}"
OUT="$PROJECT_ROOT/results/cloudedge_aligned/raw/dmax_${d_max}/${suite}/seed_${seed}"
mkdir -p "$OUT" "$PROJECT_ROOT/logs"

echo "Evaluating CloudEdge Aligned: suite=$suite, d_max=$d_max, seed=$seed"
echo "Checkpoint: $CHECKPOINT"
echo "Output: $OUT"

lerobot-eval \
  --policy.path="$CHECKPOINT" \
  --policy.device=cuda \
  --policy.eval_delay_max="$d_max" \
  --policy.use_edge_vision=true \
  --env.type=libero \
  --env.task="$suite" \
  --env.control_mode=relative \
  --env.init_states=true \
  --env.max_parallel_tasks=1 \
  --eval.batch_size=10 \
  --eval.n_episodes=50 \
  --seed="$seed" \
  --output_dir="$OUT"

echo "EVAL_COMPLETE: suite=$suite, d_max=$d_max, seed=$seed"

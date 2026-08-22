#!/bin/bash
#SBATCH --job-name=cepi05_eval
#SBATCH --output=/n/netscratch/ydu_lab/Lab/wxiong/EdgeCloud-VLA/logs/eval_cloudedge_%A_%a.out
#SBATCH --error=/n/netscratch/ydu_lab/Lab/wxiong/EdgeCloud-VLA/logs/eval_cloudedge_%A_%a.err
#SBATCH --partition=kempner
#SBATCH --account=kempner_ydu_lab
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --array=0-47

set -euo pipefail
export PROJECT_ROOT=/n/netscratch/ydu_lab/Lab/wxiong/EdgeCloud-VLA
export HF_HOME="$PROJECT_ROOT/cache/huggingface"
export HF_LEROBOT_HOME="$PROJECT_ROOT/cache/lerobot"
export MUJOCO_GL=egl
export PYTHONNOUSERSITE=1
source "$PROJECT_ROOT/.venv/bin/activate"

DELAYS=(0 10 20 40)
SUITES=(libero_spatial libero_object libero_goal libero_10)
SEEDS=(7 8 9)
delay=${DELAYS[$((SLURM_ARRAY_TASK_ID % 4))]}
suite=${SUITES[$(((SLURM_ARRAY_TASK_ID / 4) % 4))]}
seed=${SEEDS[$((SLURM_ARRAY_TASK_ID / 16))]}
CHECKPOINT="$PROJECT_ROOT/outputs/cloudedge_pi05_libero_120k/checkpoints/120000/pretrained_model"
OUT="$PROJECT_ROOT/results/raw/cloudedge_d${delay}_${suite}_s${seed}"

lerobot-eval \
  --policy.path="$CHECKPOINT" \
  --policy.device=cuda \
  --policy.eval_delay_max="$delay" \
  --policy.use_edge_vision=true \
  --policy.n_action_steps=1 \
  --env.type=libero \
  --env.task="$suite" \
  --env.control_mode=relative \
  --env.init_states=true \
  --env.max_parallel_tasks=1 \
  --eval.batch_size=5 \
  --eval.n_episodes=50 \
  --seed="$seed" \
  --output_dir="$OUT"

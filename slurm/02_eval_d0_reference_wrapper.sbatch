#!/bin/bash
#SBATCH --job-name=pi05_d0_refwrap
#SBATCH --output=/n/netscratch/ydu_lab/Lab/wxiong/EdgeCloud-VLA/logs/eval_d0_refwrap_%A_%a.out
#SBATCH --error=/n/netscratch/ydu_lab/Lab/wxiong/EdgeCloud-VLA/logs/eval_d0_refwrap_%A_%a.err
#SBATCH --partition=kempner_h200
#SBATCH --account=kempner_ydu_lab
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:nvidia_h200:1
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --array=0-3

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

SUITES=(libero_spatial libero_object libero_goal libero_10)
suite=${SUITES[$SLURM_ARRAY_TASK_ID]}
CHECKPOINT="$PROJECT_ROOT/artifacts/cloudedge_pi05_finetuned_v044_reference"
OUT="$PROJECT_ROOT/results/diagnostics/d0_finetuned_v044_wrapper_${suite}_s7"

test -s "$CHECKPOINT/model.safetensors"
test "$(stat -c %s "$CHECKPOINT/model.safetensors")" -eq 7473096344

lerobot-eval \
  --policy.path="$CHECKPOINT" \
  --policy.device=cuda \
  --policy.eval_delay_max=0 \
  --policy.use_edge_vision=false 
  --policy.compile_model=false \
  --policy.n_action_steps=10 \
  --env.type=libero \
  --env.task="$suite" \
  --env.control_mode=relative \
  --env.init_states=true \
  --env.max_parallel_tasks=1 \
  --eval.batch_size=10 \
  --eval.n_episodes=10 \
  --seed=7 \
  --output_dir="$OUT"

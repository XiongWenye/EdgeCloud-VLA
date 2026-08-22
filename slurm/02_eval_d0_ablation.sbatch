#!/bin/bash
#SBATCH --job-name=pi05_d0_ablate
#SBATCH --output=/n/netscratch/ydu_lab/Lab/wxiong/EdgeCloud-VLA/logs/eval_d0_ablate_%A_%a.out
#SBATCH --error=/n/netscratch/ydu_lab/Lab/wxiong/EdgeCloud-VLA/logs/eval_d0_ablate_%A_%a.err
#SBATCH --partition=kempner_h200
#SBATCH --account=kempner_ydu_lab
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:nvidia_h200:1
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --array=0-7

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

VARIANTS=(vanilla_meanstd cloudedge_meanstd)
SUITES=(libero_spatial libero_object libero_goal libero_10)
variant=${VARIANTS[$((SLURM_ARRAY_TASK_ID / 4))]}
suite=${SUITES[$((SLURM_ARRAY_TASK_ID % 4))]}

if [[ "$variant" == "vanilla_meanstd" ]]; then
  CHECKPOINT="$PROJECT_ROOT/artifacts/pi05_libero_base_meanstd"
  EXTRA_ARGS=()
else
  CHECKPOINT="$PROJECT_ROOT/artifacts/cloudedge_pi05_bootstrap_meanstd"
  EXTRA_ARGS=(--policy.eval_delay_max=0 --policy.use_edge_vision=false)
fi
OUT="$PROJECT_ROOT/results/diagnostics/d0_${variant}_${suite}_s7"

lerobot-eval \
  --policy.path="$CHECKPOINT" \
  --policy.device=cuda \
  --policy.n_action_steps=10 \
  "${EXTRA_ARGS[@]}" \
  --env.type=libero \
  --env.task="$suite" \
  --env.control_mode=relative \
  --env.init_states=true \
  --env.max_parallel_tasks=1 \
  --eval.batch_size=10 \
  --eval.n_episodes=10 \
  --seed=7 \
  --output_dir="$OUT"

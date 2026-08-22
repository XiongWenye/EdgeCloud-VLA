#!/bin/bash
#SBATCH --job-name=cepi05_evalsmoke
#SBATCH --output=/n/netscratch/ydu_lab/Lab/wxiong/EdgeCloud-VLA/logs/eval_smoke_%j.out
#SBATCH --error=/n/netscratch/ydu_lab/Lab/wxiong/EdgeCloud-VLA/logs/eval_smoke_%j.err
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
export MUJOCO_GL=egl
export PYTHONNOUSERSITE=1
source "$PROJECT_ROOT/.venv/bin/activate"

CHECKPOINT="$PROJECT_ROOT/artifacts/cloudedge_pi05_smoke"
test -s "$CHECKPOINT/model.safetensors"

for EDGE_MODE in false true; do
  OUT="$PROJECT_ROOT/results/smoke/edge_${EDGE_MODE}_${SLURM_JOB_ID}"
  lerobot-eval \
    --policy.path="$CHECKPOINT" \
    --policy.device=cuda \
    --policy.eval_delay_max=2 \
    --policy.use_edge_vision="$EDGE_MODE" \
    --policy.n_action_steps=1 \
    --env.type=libero \
    --env.task=libero_spatial \
    --env.control_mode=relative \
    --env.init_states=true \
    --env.max_parallel_tasks=1 \
    --eval.batch_size=1 \
    --eval.n_episodes=2 \
    --seed=7 \
    --output_dir="$OUT"
  find "$OUT" -type f -size +0c | grep -q .
done

echo "EVAL_SMOKE_SUCCESS=true"

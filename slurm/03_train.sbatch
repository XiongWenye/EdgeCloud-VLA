#!/bin/bash
#SBATCH --job-name=cepi05_120k
#SBATCH --output=/n/netscratch/ydu_lab/Lab/wxiong/EdgeCloud-VLA/logs/train_%j.out
#SBATCH --error=/n/netscratch/ydu_lab/Lab/wxiong/EdgeCloud-VLA/logs/train_%j.err
#SBATCH --partition=kempner
#SBATCH --account=kempner_ydu_lab
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH --mem=192G
#SBATCH --time=12:00:00
#SBATCH --requeue
#SBATCH --signal=B:USR1@300

set -euo pipefail
export PROJECT_ROOT=/n/netscratch/ydu_lab/Lab/wxiong/EdgeCloud-VLA
export HF_HOME="$PROJECT_ROOT/cache/huggingface"
export HF_LEROBOT_HOME="$PROJECT_ROOT/cache/lerobot"
export MUJOCO_GL=egl
export WANDB_MODE=disabled
export PYTHONNOUSERSITE=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
source "$PROJECT_ROOT/.venv/bin/activate"

RUN_DIR="$PROJECT_ROOT/outputs/cloudedge_pi05_libero_120k"
LAST_CONFIG="$RUN_DIR/checkpoints/last/pretrained_model/train_config.json"

requeue_job() {
    if [[ -n "${TRAIN_PID:-}" ]]; then
        kill -TERM "$TRAIN_PID" 2>/dev/null || true
        wait "$TRAIN_PID" 2>/dev/null || true
    fi
    scontrol requeue "$SLURM_JOB_ID"
    exit 0
}
trap requeue_job USR1

if [[ -f "$LAST_CONFIG" ]]; then
    accelerate launch --num_processes=4 -m lerobot.scripts.lerobot_train \
      --config_path="$LAST_CONFIG" --resume=true &
else
    accelerate launch --num_processes=4 -m lerobot.scripts.lerobot_train \
      --policy.path="$PROJECT_ROOT/artifacts/cloudedge_pi05_bootstrap" \
      --policy.device=cuda \
      --policy.dtype=bfloat16 \
      --policy.freeze_vision_encoder=true \
      --policy.train_expert_only=false \
      --policy.gradient_checkpointing=true \
      --policy.compile_model=false \
      --policy.push_to_hub=false \
      --dataset.repo_id=lerobot/libero \
      --dataset.revision=a1aaacb7f6cd6ee5fb43120f673cebb0cfea7dd4 \
      --dataset.video_backend=pyav \
      --seed=7 \
      --steps=120000 \
      --batch_size=2 \
      --num_workers=4 \
      --save_checkpoint=true \
      --save_freq=5000 \
      --log_freq=100 \
      --eval_freq=0 \
      --wandb.enable=false \
      --output_dir="$RUN_DIR" \
      --job_name=cloudedge_pi05_libero_120k &
fi
TRAIN_PID=$!
wait "$TRAIN_PID"

test -s "$RUN_DIR/checkpoints/120000/pretrained_model/model.safetensors"
echo "TRAINING_SUCCESS=true"


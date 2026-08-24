#!/bin/bash
#SBATCH --job-name=cepi05_v2_cand
#SBATCH --output=/n/netscratch/ydu_lab/Lab/wxiong/EdgeCloud-VLA/logs/v2_candidate_%A_%a.out
#SBATCH --error=/n/netscratch/ydu_lab/Lab/wxiong/EdgeCloud-VLA/logs/v2_candidate_%A_%a.err
#SBATCH --partition=kempner_h200
#SBATCH --account=kempner_ydu_lab
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:nvidia_h200:4
#SBATCH --mem=256G
#SBATCH --time=06:00:00
#SBATCH --array=0-1%2

set -euo pipefail
PROJECT_ROOT=/n/netscratch/ydu_lab/Lab/wxiong/EdgeCloud-VLA
CODE_ROOT=${CODE_ROOT:-$PROJECT_ROOT}
export HF_HOME=$PROJECT_ROOT/cache/huggingface
export HF_LEROBOT_HOME=$PROJECT_ROOT/cache/lerobot
export WANDB_DIR=$PROJECT_ROOT/cache/wandb
export WANDB_CACHE_DIR=$PROJECT_ROOT/cache/wandb
export WANDB_DATA_DIR=$PROJECT_ROOT/cache/wandb
export WANDB_CONFIG_DIR=$PROJECT_ROOT/cache/wandb
export TMPDIR=$PROJECT_ROOT/cache/tmp
export PIP_CACHE_DIR=$PROJECT_ROOT/cache/pip
export PYTHONNOUSERSITE=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH=$CODE_ROOT/src:${PYTHONPATH:-}
mkdir -p $PROJECT_ROOT/cache/wandb $PROJECT_ROOT/cache/tmp $PROJECT_ROOT/logs
source $PROJECT_ROOT/.venv/bin/activate
cd $CODE_ROOT

case $SLURM_ARRAY_TASK_ID in
  0)
    label=lr5e-6
    peak_lr=5e-6
    decay_lr=5e-7
    ;;
  1)
    label=lr1e-5
    peak_lr=1e-5
    decay_lr=1e-6
    ;;
  *)
    echo ERROR_BAD_ARRAY_INDEX
    exit 2
    ;;
esac

BOOTSTRAP=$PROJECT_ROOT/artifacts/cloudedge_pi05_v2_bootstrap_official_aligned
RUN_DIR=$PROJECT_ROOT/outputs/cloudedge_pi05_v2_$label
test -s $BOOTSTRAP/model.safetensors
test ! -e $RUN_DIR

accelerate launch --num_processes=4 -m lerobot.scripts.lerobot_train \
  --policy.path=$BOOTSTRAP \
  --policy.device=cuda \
  --policy.dtype=bfloat16 \
  --policy.freeze_vision_encoder=true \
  --policy.train_expert_only=false \
  --policy.gradient_checkpointing=false \
  --policy.compile_model=false \
  --policy.optimizer_lr=$peak_lr \
  --policy.scheduler_warmup_steps=500 \
  --policy.scheduler_decay_steps=10000 \
  --policy.scheduler_decay_lr=$decay_lr \
  --policy.stale_loss_weight_max=0.5 \
  --policy.stale_loss_warmup_steps=5000 \
  --policy.push_to_hub=false \
  --peft.method_type=LORA \
  --peft.r=16 \
  --dataset.repo_id=lerobot/libero \
  --dataset.revision=a1aaacb7f6cd6ee5fb43120f673cebb0cfea7dd4 \
  --dataset.video_backend=pyav \
  --seed=7 \
  --steps=10000 \
  --batch_size=4 \
  --num_workers=4 \
  --save_checkpoint=true \
  --save_freq=2500 \
  --log_freq=50 \
  --eval_freq=0 \
  --wandb.enable=true \
  --wandb.disable_artifact=true \
  --wandb.project=cloudedge_pi05_v2 \
  --output_dir=$RUN_DIR \
  --job_name=cloudedge_pi05_v2_$label

test -s $RUN_DIR/checkpoints/010000/pretrained_model/adapter_model.safetensors
echo CLOUDEDGE_V2_CANDIDATE_SUCCESS label=$label

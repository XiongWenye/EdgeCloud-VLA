#!/bin/bash
#SBATCH --job-name=cepi05_v2_e0
#SBATCH --output=/n/netscratch/ydu_lab/Lab/wxiong/EdgeCloud-VLA/logs/v2_edgezero_%A_%a.out
#SBATCH --error=/n/netscratch/ydu_lab/Lab/wxiong/EdgeCloud-VLA/logs/v2_edgezero_%A_%a.err
#SBATCH --partition=kempner_h200
#SBATCH --account=kempner_ydu_lab
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:nvidia_h200:1
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --array=0-15%8

set -euo pipefail
PROJECT_ROOT=/n/netscratch/ydu_lab/Lab/wxiong/EdgeCloud-VLA
CODE_ROOT=${CODE_ROOT:-$PROJECT_ROOT}
export HF_HOME=$PROJECT_ROOT/cache/huggingface
export HF_LEROBOT_HOME=$PROJECT_ROOT/cache/lerobot
export LIBERO_CONFIG_PATH=$PROJECT_ROOT/configs/libero
export MUJOCO_GL=egl
export PYTHONNOUSERSITE=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH=$CODE_ROOT/src:${PYTHONPATH:-}
source $PROJECT_ROOT/.venv/bin/activate
cd $CODE_ROOT

SUITES=(libero_spatial libero_object libero_goal libero_10)
DELAYS=(0 10 20 40)
delay_idx=$((SLURM_ARRAY_TASK_ID % 4))
suite_idx=$((SLURM_ARRAY_TASK_ID / 4))
suite=${SUITES[$suite_idx]}
delay=${DELAYS[$delay_idx]}
seed=7

CHECKPOINT=$PROJECT_ROOT/outputs/cloudedge_pi05_v2_lr5e-6/checkpoints/002500/pretrained_model
OUTPUT=$PROJECT_ROOT/results/cloudedge_pi05_v2_edgezero/raw/dmax_$delay/$suite/seed_$seed
test -s $CHECKPOINT/adapter_model.safetensors
test ! -e $OUTPUT
mkdir -p $OUTPUT

lerobot-eval \
  --policy.path=$CHECKPOINT \
  --policy.device=cuda \
  --policy.eval_delay_max=$delay \
  --policy.use_edge_vision=false \
  --env.type=libero \
  --env.task=$suite \
  --env.control_mode=relative \
  --env.init_states=true \
  --env.max_parallel_tasks=1 \
  --eval.batch_size=10 \
  --eval.n_episodes=50 \
  --seed=$seed \
  --output_dir=$OUTPUT

test -s $OUTPUT/eval_info.json
python -c "import json; p='$OUTPUT/eval_info.json'; x=json.load(open(p)); assert x['overall']['n_episodes'] == 500"
echo CLOUDEDGE_V2_EDGEZERO_SUCCESS suite=$suite delay=$delay seed=$seed

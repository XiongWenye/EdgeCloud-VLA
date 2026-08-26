#!/bin/bash
#SBATCH --job-name=cepi05_v2_eval
#SBATCH --output=/n/netscratch/ydu_lab/Lab/wxiong/EdgeCloud-VLA/logs/v2_candidate_eval_%A_%a.out
#SBATCH --error=/n/netscratch/ydu_lab/Lab/wxiong/EdgeCloud-VLA/logs/v2_candidate_eval_%A_%a.err
#SBATCH --partition=kempner_h200
#SBATCH --account=kempner_ydu_lab
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:nvidia_h200:1
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#SBATCH --array=0-11%8

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

delay_idx=$((SLURM_ARRAY_TASK_ID % 2))
step_idx=$(((SLURM_ARRAY_TASK_ID / 2) % 3))
candidate_idx=$((SLURM_ARRAY_TASK_ID / 6))

if [[ $delay_idx -eq 0 ]]; then delay=0; else delay=10; fi
case $step_idx in
  0) step=002500 ;;
  1) step=005000 ;;
  2) step=010000 ;;
  *) exit 2 ;;
esac
case $candidate_idx in
  0) label=lr5e-6 ;;
  1) label=lr1e-5 ;;
  *) exit 2 ;;
esac

CHECKPOINT=$PROJECT_ROOT/outputs/cloudedge_pi05_v2_$label/checkpoints/$step/pretrained_model
OUTPUT=$PROJECT_ROOT/results/cloudedge_pi05_v2_candidates/$label/step_$step/dmax_$delay/libero_spatial/seed_7
test -s $CHECKPOINT/adapter_model.safetensors
test ! -e $OUTPUT
mkdir -p $OUTPUT

lerobot-eval \
  --policy.path=$CHECKPOINT \
  --policy.device=cuda \
  --policy.eval_delay_max=$delay \
  --policy.use_edge_vision=true \
  --env.type=libero \
  --env.task=libero_spatial \
  --env.control_mode=relative \
  --env.init_states=true \
  --env.max_parallel_tasks=1 \
  --eval.batch_size=5 \
  --eval.n_episodes=10 \
  --seed=7 \
  --output_dir=$OUTPUT

test -s $OUTPUT/eval_info.json
echo CLOUDEDGE_V2_CANDIDATE_EVAL_SUCCESS label=$label step=$step delay=$delay

#!/bin/bash
#SBATCH --job-name=ce_v1_sweep
#SBATCH --output=/n/netscratch/ydu_lab/Lab/wxiong/EdgeCloud-VLA/logs/ce_v1_sweep_%A_%a.out
#SBATCH --error=/n/netscratch/ydu_lab/Lab/wxiong/EdgeCloud-VLA/logs/ce_v1_sweep_%A_%a.err
#SBATCH --partition=kempner_h200
#SBATCH --account=kempner_ydu_lab
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:nvidia_h200:1
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --array=0-11%4

set -euo pipefail
PROJECT_ROOT=/n/netscratch/ydu_lab/Lab/wxiong/EdgeCloud-VLA
export HF_HOME=$PROJECT_ROOT/cache/huggingface
export HF_LEROBOT_HOME=$PROJECT_ROOT/cache/lerobot
export LIBERO_CONFIG_PATH=$PROJECT_ROOT/configs/libero
export MUJOCO_GL=egl
export PYTHONNOUSERSITE=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
source $PROJECT_ROOT/.venv/bin/activate

delay_idx=$((SLURM_ARRAY_TASK_ID % 2))
step_idx=$((SLURM_ARRAY_TASK_ID / 2))
if [[ $delay_idx -eq 0 ]]; then delay=0; else delay=10; fi
case $step_idx in
  0) step=005000 ;;
  1) step=020000 ;;
  2) step=040000 ;;
  3) step=060000 ;;
  4) step=080000 ;;
  5) step=120000 ;;
  *) echo ERROR_BAD_STEP_INDEX; exit 2 ;;
esac

checkpoint=$PROJECT_ROOT/outputs/cloudedge_pi05_libero_aligned_120k/checkpoints/$step/pretrained_model
output=$PROJECT_ROOT/results/diagnostics_v1_checkpoint_sweep/step_$step/dmax_$delay/libero_spatial/seed_7

if [[ -f $output/eval_info.json ]]; then
  echo CHECKPOINT_SWEEP_ALREADY_COMPLETE step=$step delay=$delay
  exit 0
fi
if [[ -e $output ]]; then
  echo ERROR_PARTIAL_OUTPUT_EXISTS path=$output
  exit 2
fi
mkdir -p $output $PROJECT_ROOT/logs

echo CHECKPOINT_SWEEP_START step=$step delay=$delay checkpoint=$checkpoint
lerobot-eval --policy.path=$checkpoint --policy.device=cuda --policy.eval_delay_max=$delay --policy.use_edge_vision=true --env.type=libero --env.task=libero_spatial --env.control_mode=relative --env.init_states=true --env.max_parallel_tasks=1 --eval.batch_size=5 --eval.n_episodes=5 --seed=7 --output_dir=$output
echo CHECKPOINT_SWEEP_COMPLETE step=$step delay=$delay

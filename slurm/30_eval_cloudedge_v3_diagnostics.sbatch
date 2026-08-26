#!/bin/bash
#SBATCH --job-name=cepi05_v3_diag
#SBATCH --output=/n/netscratch/ydu_lab/Lab/wxiong/EdgeCloud-VLA/logs/v3_diagnostic_%A_%a.out
#SBATCH --error=/n/netscratch/ydu_lab/Lab/wxiong/EdgeCloud-VLA/logs/v3_diagnostic_%A_%a.err
#SBATCH --partition=kempner_h200
#SBATCH --account=kempner_ydu_lab
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:nvidia_h200:1
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --array=0-3%4

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

variant_idx=$((SLURM_ARRAY_TASK_ID / 2))
delay_idx=$((SLURM_ARRAY_TASK_ID % 2))
if [[ $delay_idx -eq 0 ]]; then delay=0; else delay=10; fi
case $variant_idx in
  0)
    variant=bootstrap_h1
    checkpoint=$PROJECT_ROOT/artifacts/cloudedge_pi05_v3_bootstrap_official_aligned
    use_edge=true
    ;;
  1)
    variant=best_lr2e-6_step625_edgezero
    checkpoint=$PROJECT_ROOT/outputs/cloudedge_pi05_v3_lr2e-6/checkpoints/000625/pretrained_model
    use_edge=false
    ;;
  *)
    exit 2
    ;;
esac

output=$PROJECT_ROOT/results/cloudedge_pi05_v3_diagnostics/$variant/dmax_$delay/libero_spatial/seed_7
test ! -e $output
mkdir -p $output
lerobot-eval \
  --policy.path=$checkpoint \
  --policy.device=cuda \
  --policy.eval_delay_max=$delay \
  --policy.use_edge_vision=$use_edge \
  --env.type=libero \
  --env.task=libero_spatial \
  --env.control_mode=relative \
  --env.init_states=true \
  --env.max_parallel_tasks=1 \
  --eval.batch_size=5 \
  --eval.n_episodes=10 \
  --seed=7 \
  --output_dir=$output

test -s $output/eval_info.json
echo CLOUDEDGE_V3_DIAGNOSTIC_SUCCESS variant=$variant delay=$delay

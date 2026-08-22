#!/bin/bash
#SBATCH --job-name=cepi05_bootstrap_ftref
#SBATCH --output=/n/netscratch/ydu_lab/Lab/wxiong/EdgeCloud-VLA/logs/bootstrap_ftref_%j.out
#SBATCH --error=/n/netscratch/ydu_lab/Lab/wxiong/EdgeCloud-VLA/logs/bootstrap_ftref_%j.err
#SBATCH --partition=kempner_h200
#SBATCH --account=kempner_ydu_lab
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:nvidia_h200:1
#SBATCH --mem=96G
#SBATCH --time=02:00:00

set -euo pipefail
export PROJECT_ROOT=/n/netscratch/ydu_lab/Lab/wxiong/EdgeCloud-VLA
export HF_HOME="$PROJECT_ROOT/cache/huggingface"
export HF_LEROBOT_HOME="$PROJECT_ROOT/cache/lerobot"
export PYTHONNOUSERSITE=1
source "$PROJECT_ROOT/.venv/bin/activate"

NORM_STATS="$PROJECT_ROOT/artifacts/openpi_pi05_libero_norm_stats.json"
mkdir -p "$PROJECT_ROOT/artifacts"
curl -L --fail --retry 5 --silent --show-error \
  -o "$NORM_STATS" \
  "https://storage.googleapis.com/openpi-assets/checkpoints/pi05_libero/assets/physical-intelligence/libero/norm_stats.json"
echo "b3a44bb2810436fb62917decaea58bd4d9110255df527dea21e8fd40c960bd84  $NORM_STATS" \
  | sha256sum --check --strict

nvidia-smi --query-gpu=index,name,uuid,memory.total --format=csv,noheader
python - <<'PY'
import torch

assert torch.cuda.is_available(), "CUDA is not available inside the allocated SLURM job"
print(f"cuda_device_count={torch.cuda.device_count()}")
print(f"cuda_device_name={torch.cuda.get_device_name(0)}")
torch.ones(1, device="cuda")
print("CUDA_PREFLIGHT_SUCCESS=true")
PY

python "$PROJECT_ROOT/scripts/bootstrap_checkpoint.py" \
  --base=/n/netscratch/ydu_lab/Lab/wxiong/EdgeCloud-VLA/artifacts/pi05_libero_finetuned_v044_reference \
  --base-revision=a217bfd3b14673cf2ce597e69997ab21866438dd \
  --norm-stats="$NORM_STATS" \
  --output="$PROJECT_ROOT/artifacts/cloudedge_pi05_finetuned_v044_reference" \
  --device=cuda \
  --history-window=21 \
  --lambda-max=0.5 \
  --warmup-steps=10000
cp "/n/netscratch/ydu_lab/Lab/wxiong/EdgeCloud-VLA/artifacts/pi05_libero_finetuned_v044_reference/policy_preprocessor_step_2_normalizer_processor.safetensors" 
  "/n/netscratch/ydu_lab/Lab/wxiong/EdgeCloud-VLA/artifacts/cloudedge_pi05_finetuned_v044_reference/policy_preprocessor_step_3_normalizer_processor.safetensors"
cp "/n/netscratch/ydu_lab/Lab/wxiong/EdgeCloud-VLA/artifacts/pi05_libero_finetuned_v044_reference/policy_postprocessor_step_0_unnormalizer_processor.safetensors" 
  "/n/netscratch/ydu_lab/Lab/wxiong/EdgeCloud-VLA/artifacts/cloudedge_pi05_finetuned_v044_reference/policy_postprocessor_step_0_unnormalizer_processor.safetensors"
cmp "/n/netscratch/ydu_lab/Lab/wxiong/EdgeCloud-VLA/artifacts/pi05_libero_finetuned_v044_reference/policy_preprocessor_step_2_normalizer_processor.safetensors" 
  "/n/netscratch/ydu_lab/Lab/wxiong/EdgeCloud-VLA/artifacts/cloudedge_pi05_finetuned_v044_reference/policy_preprocessor_step_3_normalizer_processor.safetensors"
cmp "/n/netscratch/ydu_lab/Lab/wxiong/EdgeCloud-VLA/artifacts/pi05_libero_finetuned_v044_reference/policy_postprocessor_step_0_unnormalizer_processor.safetensors" 
  "/n/netscratch/ydu_lab/Lab/wxiong/EdgeCloud-VLA/artifacts/cloudedge_pi05_finetuned_v044_reference/policy_postprocessor_step_0_unnormalizer_processor.safetensors"
echo "BOOTSTRAP_FINETUNED_REFERENCE_SUCCESS=true"

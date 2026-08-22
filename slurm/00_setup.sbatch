#!/bin/bash
#SBATCH --job-name=cepi05_setup
#SBATCH --output=/n/netscratch/ydu_lab/Lab/wxiong/EdgeCloud-VLA/logs/setup_%j.out
#SBATCH --error=/n/netscratch/ydu_lab/Lab/wxiong/EdgeCloud-VLA/logs/setup_%j.err
#SBATCH --partition=kempner
#SBATCH --account=kempner_ydu_lab
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH --time=04:00:00

set -euo pipefail

export PROJECT_ROOT=/n/netscratch/ydu_lab/Lab/wxiong/EdgeCloud-VLA
export ENV_PREFIX="$PROJECT_ROOT/.venv"
export LEROBOT_ROOT="$PROJECT_ROOT/third_party/lerobot"
export HF_HOME="$PROJECT_ROOT/cache/huggingface"
export HF_LEROBOT_HOME="$PROJECT_ROOT/cache/lerobot"
export PIP_CACHE_DIR="$PROJECT_ROOT/cache/pip"
export PYTHONNOUSERSITE=1
export TOKENIZERS_PARALLELISM=false

mkdir -p "$PROJECT_ROOT/logs" "$PROJECT_ROOT/cache" "$PROJECT_ROOT/third_party"
if [[ ! -x "$ENV_PREFIX/bin/python" ]]; then
    /n/sw/Mambaforge-23.11.0-0/bin/python -m venv "$ENV_PREFIX"
fi
source "$ENV_PREFIX/bin/activate"
python -m pip install --upgrade pip wheel "setuptools<81"

if [[ ! -d "$LEROBOT_ROOT/.git" ]]; then
    git clone https://github.com/huggingface/lerobot.git "$LEROBOT_ROOT"
fi
git -C "$LEROBOT_ROOT" fetch --depth 1 origin 8fff0fde7c79f23a93d845d1a50e985de01f8b8a
git -C "$LEROBOT_ROOT" checkout --detach 8fff0fde7c79f23a93d845d1a50e985de01f8b8a
python -m pip install -e "$LEROBOT_ROOT"
python -m pip install "cmake>=3.29,<4"
python -m pip install --no-build-isolation "hf-libero==0.1.4" "scipy>=1.10.1,<1.15"
python -m pip install --force-reinstall \
  "transformers @ git+https://github.com/huggingface/transformers.git@dcddb970176382c0fcf4521b0c0e6fc15894dfe0"
python -m pip install \
  "huggingface-hub[cli,hf-transfer]>=0.34.2,<0.36.0" \
  "fsspec<=2026.2.0" \
  "packaging>=24.2,<26" \
  "setuptools>=71,<81"
python -m pip install --no-deps -e "$PROJECT_ROOT"
python -m pip check

python - <<'PY'
from huggingface_hub import snapshot_download

snapshot_download("lerobot/pi05_libero_base", revision="a217bfd3b14673cf2ce597e69997ab21866438dd")
snapshot_download("google/siglip-base-patch16-224", revision="7fd15f0689c79d79e38b1c2e2e2370a7bf2761ed")
snapshot_download(
    "lerobot/libero",
    repo_type="dataset",
    revision="a1aaacb7f6cd6ee5fb43120f673cebb0cfea7dd4",
)
print("ARTIFACT_DOWNLOAD_SUCCESS=true")
PY

python "$PROJECT_ROOT/scripts/static_validate.py"
python - <<'PY'
import os
import torch
import lerobot
import lerobot_policy_cloudedge_pi05

print(f"lerobot={lerobot.__version__}")
print(f"torch={torch.__version__} cuda={torch.version.cuda}")
if "SLURM_JOB_ID" in os.environ:
    assert torch.cuda.is_available()
    print(f"gpu={torch.cuda.get_device_name(0)}")
else:
    print(f"login_node_cuda_available={torch.cuda.is_available()}")
print("PLUGIN_IMPORT_SUCCESS=true")
PY

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
export LIBERO_CONFIG_PATH="$PROJECT_ROOT/configs/libero"
export PIP_CACHE_DIR="$PROJECT_ROOT/cache/pip"
export PYTHONNOUSERSITE=1
export TOKENIZERS_PARALLELISM=false
export PATCH_FILE="$PROJECT_ROOT/patches/0001-align-lerobot-pi05-with-openpi-libero.patch"

mkdir -p "$PROJECT_ROOT/logs" "$PROJECT_ROOT/cache" "$PROJECT_ROOT/third_party" "$PROJECT_ROOT/patches"
if [[ ! -x "$ENV_PREFIX/bin/python" ]]; then
    /n/sw/Mambaforge-23.11.0-0/bin/python -m venv "$ENV_PREFIX"
fi
source "$ENV_PREFIX/bin/activate"
python -m pip install --upgrade pip wheel "setuptools<81"

if [[ ! -d "$LEROBOT_ROOT/.git" ]]; then
    git clone https://github.com/huggingface/lerobot.git "$LEROBOT_ROOT"
fi

# Ensure LeRobot repository state
PINNED_LEROBOT_COMMIT="8fff0fde7c79f23a93d845d1a50e985de01f8b8a"

# Check if LeRobot has the patch applied or is at pinned commit
CURRENT_LEROBOT_REV=$(git -C "$LEROBOT_ROOT" rev-parse HEAD)
echo "Current LeRobot rev: $CURRENT_LEROBOT_REV"

# If dirty, check if diff matches our patch
if [[ -n $(git -C "$LEROBOT_ROOT" status --porcelain) ]]; then
    # Test if patch applies in reverse
    if git -C "$LEROBOT_ROOT" apply --check --reverse "$PATCH_FILE" 2>/dev/null; then
        echo "LeRobot has alignment patch applied cleanly."
    else
        echo "ERROR: LeRobot working tree has uncommitted changes that do not match alignment patch."
        git -C "$LEROBOT_ROOT" status --short
        exit 1
    fi
else
    # Clean working tree. If at pinned commit, apply patch
    if [[ "$CURRENT_LEROBOT_REV" == "$PINNED_LEROBOT_COMMIT"* ]]; then
        echo "Applying alignment patch to LeRobot $PINNED_LEROBOT_COMMIT..."
        git -C "$LEROBOT_ROOT" apply "$PATCH_FILE"
    elif git -C "$LEROBOT_ROOT" log -n 1 --oneline | grep -qi "align"; then
        echo "LeRobot is on aligned commit."
    else
        # Try checking out pinned commit then applying patch
        git -C "$LEROBOT_ROOT" fetch --depth 1 origin "$PINNED_LEROBOT_COMMIT"
        git -C "$LEROBOT_ROOT" checkout --detach "$PINNED_LEROBOT_COMMIT"
        git -C "$LEROBOT_ROOT" apply "$PATCH_FILE"
    fi
fi

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

python - <<PY
from huggingface_hub import snapshot_download

snapshot_download("google/siglip-base-patch16-224", revision="7fd15f0689c79d79e38b1c2e2e2370a7bf2761ed")
snapshot_download(
    "lerobot/libero",
    repo_type="dataset",
    revision="a1aaacb7f6cd6ee5fb43120f673cebb0cfea7dd4",
)
snapshot_download(
    "lerobot/libero-assets",
    repo_type="dataset",
    revision="0b3ea86be5fe169d0fd036ae63d1070ec09e90f6",
    local_dir="/n/netscratch/ydu_lab/Lab/wxiong/EdgeCloud-VLA/cache/libero/assets",
)
print("ARTIFACT_DOWNLOAD_SUCCESS=true")
PY

LIBERO_SITE_ASSETS="$ENV_PREFIX/lib/python3.10/site-packages/libero/libero/assets"
if [[ ! -e "$LIBERO_SITE_ASSETS" ]]; then
    ln -s "$PROJECT_ROOT/cache/libero/assets" "$LIBERO_SITE_ASSETS"
fi

python -m pytest "$PROJECT_ROOT/tests/test_pi05_alignment.py" -v
python "$PROJECT_ROOT/scripts/static_validate.py"

python - <<PY
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
print("SETUP_SUCCESS=true")
PY

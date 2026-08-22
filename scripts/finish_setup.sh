#!/bin/bash
set -euo pipefail

export PROJECT_ROOT=/n/netscratch/ydu_lab/Lab/wxiong/EdgeCloud-VLA
export HF_HOME="$PROJECT_ROOT/cache/huggingface"
export HF_LEROBOT_HOME="$PROJECT_ROOT/cache/lerobot"
export LIBERO_CONFIG_PATH="$PROJECT_ROOT/configs/libero"
export PIP_CACHE_DIR="$PROJECT_ROOT/cache/pip"
export PYTHONNOUSERSITE=1
export TOKENIZERS_PARALLELISM=false
source "$PROJECT_ROOT/.venv/bin/activate"

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
snapshot_download(
    "lerobot/libero-assets",
    repo_type="dataset",
    revision="0b3ea86be5fe169d0fd036ae63d1070ec09e90f6",
    local_dir="/n/netscratch/ydu_lab/Lab/wxiong/EdgeCloud-VLA/cache/libero/assets",
)
print("ARTIFACT_DOWNLOAD_SUCCESS=true")
PY

LIBERO_SITE_ASSETS="$PROJECT_ROOT/.venv/lib/python3.10/site-packages/libero/libero/assets"
if [[ ! -e "$LIBERO_SITE_ASSETS" ]]; then
    ln -s "$PROJECT_ROOT/cache/libero/assets" "$LIBERO_SITE_ASSETS"
fi

python "$PROJECT_ROOT/scripts/static_validate.py"
python - <<'PY'
import torch
import lerobot
import lerobot_policy_cloudedge_pi05
import transformers

print(f"lerobot={lerobot.__version__}")
print(f"transformers={transformers.__version__}")
print(f"torch={torch.__version__} cuda={torch.version.cuda}")
print(f"login_node_cuda_available={torch.cuda.is_available()}")
print("PLUGIN_IMPORT_SUCCESS=true")
PY

echo "LOGIN_SETUP_SUCCESS=true"

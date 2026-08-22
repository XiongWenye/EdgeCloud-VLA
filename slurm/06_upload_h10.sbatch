#!/bin/bash
#SBATCH --job-name=cepi05_h10_upload
#SBATCH --output=/n/netscratch/ydu_lab/Lab/wxiong/EdgeCloud-VLA/logs/upload_h10_%j.out
#SBATCH --error=/n/netscratch/ydu_lab/Lab/wxiong/EdgeCloud-VLA/logs/upload_h10_%j.err
#SBATCH --partition=kempner
#SBATCH --account=kempner_ydu_lab
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=04:00:00

set -euo pipefail
export PROJECT_ROOT=/n/netscratch/ydu_lab/Lab/wxiong/EdgeCloud-VLA
export HF_HOME="$PROJECT_ROOT/cache/huggingface"
export PYTHONNOUSERSITE=1
export MODEL_REPO=${MODEL_REPO:-wxiong/pi05-cloudedge-libero}
source "$PROJECT_ROOT/.venv/bin/activate"

CHECKPOINT="$PROJECT_ROOT/outputs/cloudedge_pi05_libero_120k/checkpoints/120000/pretrained_model"
test -s "$CHECKPOINT/model.safetensors"
test -s "$PROJECT_ROOT/results/summary_h10.md"
python - "$PROJECT_ROOT/model_card.md" "$PROJECT_ROOT/results/summary_h10.md" "$CHECKPOINT/README.md" <<'PY'
import sys
from pathlib import Path

model_card, results, output = map(Path, sys.argv[1:])
protocol = (
    "Evaluation uses a 10-action execution horizon. Delays are measured in "
    "environment steps because observations are retained at every step, including "
    "while queued actions are executing."
)
output.write_text(
    model_card.read_text().rstrip()
    + "\n\n"
    + protocol
    + "\n\n"
    + results.read_text()
)
PY
hf auth whoami
hf upload-large-folder "$MODEL_REPO" "$CHECKPOINT" --repo-type=model --revision=main --num-workers=4
python - "$MODEL_REPO" <<'PY'
import sys
from huggingface_hub import HfApi

info = HfApi().model_info(sys.argv[1], files_metadata=True)
names = {item.rfilename for item in info.siblings}
required = {"config.json", "model.safetensors", "policy_preprocessor.json", "policy_postprocessor.json"}
assert required <= names, (required - names)
print(f"repo={info.id} commit={info.sha}")
print("UPLOAD_VALIDATION_SUCCESS=true")
PY

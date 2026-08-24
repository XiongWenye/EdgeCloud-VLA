#!/bin/bash
#SBATCH --job-name=ce_diag
#SBATCH --output=/n/netscratch/ydu_lab/Lab/wxiong/EdgeCloud-VLA/logs/mechanism_diagnostic_%j.out
#SBATCH --error=/n/netscratch/ydu_lab/Lab/wxiong/EdgeCloud-VLA/logs/mechanism_diagnostic_%j.err
#SBATCH --partition=kempner_h200
#SBATCH --account=kempner_ydu_lab
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=01:00:00

set -euo pipefail
export PROJECT_ROOT=/n/netscratch/ydu_lab/Lab/wxiong/EdgeCloud-VLA
export HF_HOME="$PROJECT_ROOT/cache/huggingface"
export HF_LEROBOT_HOME="$PROJECT_ROOT/cache/lerobot"
export WANDB_DIR="$PROJECT_ROOT/cache/wandb"
export TMPDIR="$PROJECT_ROOT/cache/tmp"
export PYTHONNOUSERSITE=1

source "$PROJECT_ROOT/.venv/bin/activate"
mkdir -p "$PROJECT_ROOT/logs" "$PROJECT_ROOT/results"

python "$PROJECT_ROOT/scripts/run_mechanism_diagnostic.py"

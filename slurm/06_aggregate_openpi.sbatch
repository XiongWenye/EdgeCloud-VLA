#!/bin/bash
#SBATCH --job-name=cepi05_openpi_aggregate
#SBATCH --output=/n/netscratch/ydu_lab/Lab/wxiong/EdgeCloud-VLA/logs/aggregate_openpi_%j.out
#SBATCH --error=/n/netscratch/ydu_lab/Lab/wxiong/EdgeCloud-VLA/logs/aggregate_openpi_%j.err
#SBATCH --partition=kempner
#SBATCH --account=kempner_ydu_lab
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=01:00:00

set -euo pipefail
export PROJECT_ROOT=/n/netscratch/ydu_lab/Lab/wxiong/EdgeCloud-VLA
export PYTHONNOUSERSITE=1
source "$PROJECT_ROOT/.venv/bin/activate"

python "$PROJECT_ROOT/scripts/aggregate_results.py" \
  "$PROJECT_ROOT/results/raw_openpi_h5" \
  --output "$PROJECT_ROOT/results/summary_openpi_h5.csv" \
  --markdown-output "$PROJECT_ROOT/results/summary_openpi_h5.md" \
  --validate-paper-grid

test -s "$PROJECT_ROOT/results/summary_openpi_h5.csv"
test -s "$PROJECT_ROOT/results/summary_openpi_h5.md"
echo "AGGREGATION_OPENPI_SUCCESS=true"

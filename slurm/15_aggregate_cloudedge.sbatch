#!/bin/bash
#SBATCH --job-name=cepi05_aggregate
#SBATCH --output=/n/netscratch/ydu_lab/Lab/wxiong/EdgeCloud-VLA/logs/aggregate_%j.out
#SBATCH --error=/n/netscratch/ydu_lab/Lab/wxiong/EdgeCloud-VLA/logs/aggregate_%j.err
#SBATCH --partition=kempner
#SBATCH --account=kempner_ydu_lab
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00

set -euo pipefail
export PROJECT_ROOT=/n/netscratch/ydu_lab/Lab/wxiong/EdgeCloud-VLA
source "$PROJECT_ROOT/.venv/bin/activate"

# 1. Aggregate core 64-record matrix (d_max in [0, 10, 20, 40])
python "$PROJECT_ROOT/scripts/aggregate_cloudedge_results.py" \
  --ce-dir="$PROJECT_ROOT/results/cloudedge_aligned/raw" \
  --base-dir="$PROJECT_ROOT/results/baseline_aligned_delayed/raw" \
  --output-json="$PROJECT_ROOT/results/cloudedge_aligned_summary.json" \
  --output-md="$PROJECT_ROOT/results/cloudedge_aligned_summary.md"

# 2. Aggregate full 8-point delay curve (d_max in [0, 5, 10, 15, 20, 25, 30, 40])
python "$PROJECT_ROOT/scripts/aggregate_cloudedge_results.py" \
  --ce-dir="$PROJECT_ROOT/results/cloudedge_aligned/raw" \
  --base-dir="$PROJECT_ROOT/results/baseline_aligned_delayed/raw" \
  --output-json="$PROJECT_ROOT/results/cloudedge_aligned_full_curve_summary.json" \
  --output-md="$PROJECT_ROOT/results/cloudedge_aligned_full_curve_summary.md" \
  --full-curve

echo "ALL_AGGREGATION_COMPLETE=true"

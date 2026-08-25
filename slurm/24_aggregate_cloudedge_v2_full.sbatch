#!/bin/bash
#SBATCH --job-name=cepi05_v2_sum
#SBATCH --output=/n/netscratch/ydu_lab/Lab/wxiong/EdgeCloud-VLA/logs/v2_full_summary_%j.out
#SBATCH --error=/n/netscratch/ydu_lab/Lab/wxiong/EdgeCloud-VLA/logs/v2_full_summary_%j.err
#SBATCH --partition=shared
#SBATCH --account=ydu_lab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:15:00

set -euo pipefail
PROJECT_ROOT=/n/netscratch/ydu_lab/Lab/wxiong/EdgeCloud-VLA
CODE_ROOT=${CODE_ROOT:-$PROJECT_ROOT}
source $PROJECT_ROOT/.venv/bin/activate
cd $PROJECT_ROOT

python $CODE_ROOT/scripts/aggregate_cloudedge_results.py \
  --ce-dir $PROJECT_ROOT/results/cloudedge_pi05_v2_full/raw \
  --base-dir $PROJECT_ROOT/results/baseline_aligned_delayed/raw \
  --output-json $PROJECT_ROOT/results/cloudedge_pi05_v2_full_summary.json \
  --output-md $PROJECT_ROOT/results/cloudedge_pi05_v2_full_summary.md
python $CODE_ROOT/scripts/aggregate_v2_edge_ablation.py
test -s $PROJECT_ROOT/results/cloudedge_pi05_v2_full_summary.json
test -s $PROJECT_ROOT/results/cloudedge_pi05_v2_edge_ablation_summary.json
echo CLOUDEDGE_V2_FULL_AGGREGATE_SUCCESS=true

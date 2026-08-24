#!/bin/bash
#SBATCH --job-name=cepi05_v2_gate
#SBATCH --output=/n/netscratch/ydu_lab/Lab/wxiong/EdgeCloud-VLA/logs/v2_candidate_gate_%j.out
#SBATCH --error=/n/netscratch/ydu_lab/Lab/wxiong/EdgeCloud-VLA/logs/v2_candidate_gate_%j.err
#SBATCH --partition=kempner
#SBATCH --account=kempner_ydu_lab
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
python $CODE_ROOT/scripts/aggregate_v2_candidates.py
cat $PROJECT_ROOT/results/cloudedge_pi05_v2_candidates_summary.md

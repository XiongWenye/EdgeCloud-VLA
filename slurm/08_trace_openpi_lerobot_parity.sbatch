#!/bin/bash
#SBATCH --job-name=pi05_parity
#SBATCH --output=/n/netscratch/ydu_lab/Lab/wxiong/EdgeCloud-VLA/logs/pi05_parity_%j.out
#SBATCH --error=/n/netscratch/ydu_lab/Lab/wxiong/EdgeCloud-VLA/logs/pi05_parity_%j.err
#SBATCH --partition=kempner_h200
#SBATCH --account=kempner_ydu_lab
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:nvidia_h200:1
#SBATCH --mem=128G
#SBATCH --time=04:00:00

set -euo pipefail

PROJECT_ROOT=/n/netscratch/ydu_lab/Lab/wxiong/EdgeCloud-VLA
OPENPI_ROOT="$PROJECT_ROOT/reference/openpi"
CLIENT_PY="$OPENPI_ROOT/examples/libero/.venv/bin/python"
OPENPI_PY="$PROJECT_ROOT/reference/openpi_server_env_pip/bin/python"
LEROBOT_PY="$PROJECT_ROOT/.venv/bin/python"
CHECKPOINT="$PROJECT_ROOT/cache/openpi_reference/openpi-assets/checkpoints/pi05_libero"
OUT="$PROJECT_ROOT/results/parity"
OBS="$OUT/openpi_observation_spatial_t0_e0.npz"
OPENPI_TRACE="$OUT/openpi_trace_spatial_t0_e0.npz"
LEROBOT_TRACE="$OUT/lerobot_trace_spatial_t0_e0.npz"

mkdir -p "$OUT" "$PROJECT_ROOT/logs"
export MUJOCO_GL=egl
export PYTHONNOUSERSITE=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export OPENPI_DATA_HOME="$PROJECT_ROOT/cache/openpi_reference"
export LIBERO_CONFIG_PATH="$PROJECT_ROOT/config/openpi_reference/libero"

export PYTHONPATH="$OPENPI_ROOT/third_party/libero:$OPENPI_ROOT/packages/openpi-client/src"
"$CLIENT_PY" "$PROJECT_ROOT/scripts/capture_openpi_libero_observation.py"     --output "$OBS"     --suite libero_spatial     --task-id 0     --episode-id 0     --seed 7

export PYTHONPATH="$OPENPI_ROOT/src:$OPENPI_ROOT/packages/openpi-client/src"
"$OPENPI_PY" "$PROJECT_ROOT/scripts/trace_openpi_reference.py"     --observation "$OBS"     --checkpoint "$CHECKPOINT"     --output "$OPENPI_TRACE"

unset PYTHONPATH
"$LEROBOT_PY" "$PROJECT_ROOT/scripts/trace_lerobot_conversion.py"     --observation "$OBS"     --openpi-trace "$OPENPI_TRACE"     --checkpoint "$PROJECT_ROOT/artifacts/pi05_libero_base_openpi_norm"     --tokenizer "$PROJECT_ROOT/cache/openpi_reference/big_vision/paligemma_tokenizer.model"     --output "$LEROBOT_TRACE"
"$LEROBOT_PY" "$PROJECT_ROOT/scripts/compare_openpi_lerobot_trace.py"     --observation "$OBS"     --openpi "$OPENPI_TRACE"     --lerobot "$LEROBOT_TRACE"     --checkpoint "$PROJECT_ROOT/artifacts/pi05_libero_base_openpi_norm"     --json "$OUT/openpi_vs_lerobot.json"     --markdown "$OUT/openpi_vs_lerobot.md"

echo PI05_PARITY_TRACE_OK

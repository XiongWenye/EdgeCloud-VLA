#!/bin/bash
#SBATCH --job-name=openpi_ref_eval
#SBATCH --output=/n/netscratch/ydu_lab/Lab/wxiong/EdgeCloud-VLA/logs/openpi_ref_%A_%a.out
#SBATCH --error=/n/netscratch/ydu_lab/Lab/wxiong/EdgeCloud-VLA/logs/openpi_ref_%A_%a.err
#SBATCH --partition=kempner_h200
#SBATCH --account=kempner_ydu_lab
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:nvidia_h200:1
#SBATCH --mem=96G
#SBATCH --time=24:00:00

set -euo pipefail

PROJECT_ROOT=/n/netscratch/ydu_lab/Lab/wxiong/EdgeCloud-VLA
OPENPI_ROOT="$PROJECT_ROOT/reference/openpi"
SERVER_PY="$PROJECT_ROOT/reference/openpi_server_env_pip/bin/python"
CLIENT_PY="$OPENPI_ROOT/examples/libero/.venv/bin/python"
OPENPI_COMMIT=15a9616a00943ada6c20a0f158e3adb39df2ccac
SUITES=(libero_spatial libero_object libero_goal libero_10)
TASK_ID=${SLURM_ARRAY_TASK_ID:-0}
SUITE=${SUITES[$TASK_ID]}
NUM_TRIALS=${NUM_TRIALS:-50}
RUN_TAG=${RUN_TAG:-official_d0_50}
PORT=${PORT:-$((18000 + (SLURM_JOB_ID % 1000) * 10 + TASK_ID))}
RUN_DIR="$PROJECT_ROOT/results/openpi_reference/$RUN_TAG/$SUITE"

mkdir -p "$RUN_DIR" "$PROJECT_ROOT/logs"

test "$(git -C "$OPENPI_ROOT" rev-parse HEAD)" = "$OPENPI_COMMIT"
test -f "$PROJECT_ROOT/cache/openpi_reference/openpi-assets/checkpoints/pi05_libero/params/manifest.ocdbt"
test -x "$SERVER_PY"
test -x "$CLIENT_PY"

export OPENPI_DATA_HOME="$PROJECT_ROOT/cache/openpi_reference"
export LIBERO_CONFIG_PATH="$PROJECT_ROOT/config/openpi_reference/libero"
export MUJOCO_GL=egl
export PYTHONNOUSERSITE=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONPATH="$OPENPI_ROOT/src:$OPENPI_ROOT/packages/openpi-client/src"

cd "$OPENPI_ROOT"
"$SERVER_PY" scripts/serve_policy.py --env LIBERO --port "$PORT" >"$RUN_DIR/server.log" 2>&1 &
SERVER_PID=$!
cleanup() {
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
}
trap cleanup EXIT

for _ in $(seq 1 1800); do
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "OpenPI server exited before becoming ready" >&2
        tail -200 "$RUN_DIR/server.log" >&2
        exit 1
    fi
    if (echo >"/dev/tcp/127.0.0.1/$PORT") 2>/dev/null; then
        break
    fi
    sleep 1
done

if ! (echo >"/dev/tcp/127.0.0.1/$PORT") 2>/dev/null; then
    echo "Timed out waiting for OpenPI server on port $PORT" >&2
    exit 1
fi

export PYTHONPATH="$OPENPI_ROOT/third_party/libero:$OPENPI_ROOT/packages/openpi-client/src"
"$CLIENT_PY" examples/libero/main.py \
    --args.host 127.0.0.1 \
    --args.port "$PORT" \
    --args.task-suite-name "$SUITE" \
    --args.num-trials-per-task "$NUM_TRIALS" \
    --args.video-out-path "$RUN_DIR/videos" \
    --args.seed 7 \
    --args.resize-size 224 \
    --args.replan-steps 5 \
    --args.num-steps-wait 10 \
    2>&1 | tee "$RUN_DIR/client.log"

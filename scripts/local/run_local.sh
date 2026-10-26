#!/usr/bin/env bash
# Run one experiment on a single machine: one vLLM server plus local sandbox evaluation.
# Usage: run_local.sh CONFIG OUT_DIR [EVAL_WORKERS] [extra `ttt run` arguments...]
# The server listens on 127.0.0.1 behind a random API key, on a free port unless PORT is set; the
# run starts once the server is healthy and no other process listens on its port.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
# shellcheck source=../lib/serve.sh
source "$HERE/../lib/serve.sh"

MODEL_PATH=${MODEL_PATH:?set MODEL_PATH to the model directory or hub id}
SERVED_MODEL=${SERVED_MODEL:-model}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-32768}
POLL_S=${POLL_S:-5}
PYTHON=${PYTHON:-python}
VLLM=${VLLM:-vllm}

usage="usage: run_local.sh CONFIG OUT_DIR [EVAL_WORKERS] [ttt run args...]"
CONFIG=${1:?$usage}
OUT_DIR=${2:?$usage}
EVAL_WORKERS=${3:-8}
shift $(( $# < 3 ? $# : 3 ))
VLLM_LOG=${VLLM_LOG:-${OUT_DIR%/}.vllm.log}
die() { echo "$1" >&2; exit 1; }
# The run creates OUT_DIR itself, but the vLLM log is written first, next to it or at VLLM_LOG.
mkdir -p -- "$(dirname -- "${OUT_DIR%/}")" "$(dirname -- "$VLLM_LOG")"

if [[ -n ${PORT:-} ]]; then
  [[ -z $(listen_inodes "$PORT") ]] || die "port $PORT is already in use"
else
  PORT=$(free_ports 1)
fi
[[ -r /proc/net/tcp ]] || echo "warning: /proc/net/tcp is unreadable; port $PORT is not checked for sharing" >&2
TTT_API_KEY=$("$PYTHON" -c 'import secrets; print(secrets.token_hex(24))')
export TTT_API_KEY

VLLM_API_KEY=$TTT_API_KEY "$VLLM" serve "$MODEL_PATH" --served-model-name "$SERVED_MODEL" \
  --host 127.0.0.1 --port "$PORT" --max-model-len "$MAX_MODEL_LEN" \
  --safetensors-load-strategy lazy --enable-prefix-caching > "$VLLM_LOG" 2>&1 &
VLLM_PID=$!
trap 'kill "$VLLM_PID" 2> /dev/null || true; wait "$VLLM_PID" 2> /dev/null || true' EXIT

while :; do
  case $(probe_server "$VLLM_PID" "$PORT") in
    ready) break ;;
    exited) die "vLLM exited during startup; see $VLLM_LOG" ;;
    foreign) die "another process also listens on port $PORT; rerun to pick a new port" ;;
  esac
  sleep "$POLL_S"
done

"$PYTHON" -m ttt run --config "$CONFIG" --vllm-url "http://127.0.0.1:$PORT" \
  --served-model "$SERVED_MODEL" --out-dir "$OUT_DIR" --local-eval "$EVAL_WORKERS" \
  --api-key-env TTT_API_KEY "$@"

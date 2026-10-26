#!/bin/bash
#SBATCH --job-name=ttt-driver
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --qos=YOUR_QOS
#SBATCH --partition=YOUR_CPU_PARTITION
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=72:00:00
# The search driver: heartbeats RUN_DIR/driver.alive, waits for the vLLM job's server URLs, then runs
# `ttt run` on the pool's endpoint directory (rescanned, so workers that start later join the run).
# It gives up as soon as the vLLM job writes RUN_DIR/servers/FAILED, before or during the run, or
# when the servers are not all up after SERVER_WAIT_S (default 4 h, including the GPU queue time).
# On exit it writes RUN_DIR/driver.done so the GPU job stops (and POOL_DIR/STOP if STOP_POOL_ON_EXIT=1).
set -euo pipefail
: "${CONFIG:?}" "${RUN_DIR:?}" "${POOL_DIR:?}" "${N_SERVERS:?}"
SERVED_MODEL=${SERVED_MODEL:-model}
SERVER_WAIT_S=${SERVER_WAIT_S:-14400}
POLL_S=${POLL_S:-30}
PYTHON=${PYTHON:-python}

run_pid=
finish() {
  if [[ -n $run_pid ]] && kill -0 "$run_pid" 2> /dev/null; then
    kill "$run_pid" 2> /dev/null || true
    wait "$run_pid" 2> /dev/null || true
  fi
  touch "$RUN_DIR/driver.done"
  if [[ ${STOP_POOL_ON_EXIT:-} == 1 ]]; then touch "$POOL_DIR/STOP"; fi
}
trap finish EXIT
gpu_failed() {
  [[ -e $RUN_DIR/servers/FAILED ]] || return 1
  echo "giving up: $(< "$RUN_DIR/servers/FAILED")" >&2
}

touch "$RUN_DIR/driver.alive"
shopt -s nullglob
deadline=$(( SECONDS + SERVER_WAIT_S ))
while :; do
  if gpu_failed; then exit 1; fi
  urls=("$RUN_DIR"/servers/*.url)
  (( ${#urls[@]} < N_SERVERS )) || break
  (( SECONDS < deadline )) || { echo "giving up: the vLLM servers did not come up within ${SERVER_WAIT_S}s" >&2; exit 1; }
  sleep "$POLL_S"
  touch "$RUN_DIR/driver.alive"
done
args=()
for f in "${urls[@]}"; do args+=(--vllm-url "$(< "$f")"); done
resume=()
if [[ ${RESUME:-0} == 1 ]]; then resume=(--resume); fi

TTT_API_KEY=$(< "$RUN_DIR/api_key")
export TTT_API_KEY
"$PYTHON" -m ttt run --config "$CONFIG" "${args[@]}" --served-model "$SERVED_MODEL" \
  --out-dir "$RUN_DIR/run" --eval-workers "$POOL_DIR/endpoints" --api-key-env TTT_API_KEY \
  ${resume[@]+"${resume[@]}"} &
run_pid=$!
while kill -0 "$run_pid" 2> /dev/null; do
  if gpu_failed; then
    echo "stopping the run; continue it with RESUME=1" >&2
    exit 1
  fi
  touch "$RUN_DIR/driver.alive"
  sleep "$POLL_S"
done
rc=0
wait "$run_pid" || rc=$?
run_pid=
exit "$rc"

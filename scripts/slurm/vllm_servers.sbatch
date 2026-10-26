#!/bin/bash
#SBATCH --job-name=ttt-vllm
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --qos=YOUR_QOS
#SBATCH --partition=YOUR_GPU_PARTITION
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=4
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=72:00:00
# Generation servers: one `vllm serve` per GPUS_PER_SERVER of the job's GPUs, each on a port picked
# free at start. A server's URL goes to RUN_DIR/servers/ once it is healthy and its port is not
# shared with another process. submit_run.sh starts this job only after the driver has started; the
# job ends when the driver writes RUN_DIR/driver.done or stops heartbeating RUN_DIR/driver.alive.
# On every exit other than driver.done it writes RUN_DIR/servers/FAILED, so the driver gives up.
set -euo pipefail
: "${RUN_DIR:?}" "${MODEL_PATH:?}" "${N_SERVERS:?}" "${TTT_SCRIPTS:?set TTT_SCRIPTS (submit_run.sh exports it)}"
# shellcheck source=../lib/serve.sh
source "$TTT_SCRIPTS/lib/serve.sh"
SERVED_MODEL=${SERVED_MODEL:-model}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-32768}
GPUS_PER_SERVER=${GPUS_PER_SERVER:-1}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.92}
DRIVER_START_WAIT_S=${DRIVER_START_WAIT_S:-900}
DRIVER_STALE_S=${DRIVER_STALE_S:-600}
POLL_S=${POLL_S:-10}
VLLM=${VLLM:-vllm}

servers=$RUN_DIR/servers
job=${SLURM_JOB_ID:-local}
pids=()
scratch=
reason="stopped by an error"
on_exit() {
  local rc=$?
  if [[ ! -e $RUN_DIR/driver.done ]]; then
    write_atomic "$servers/FAILED" "vLLM job $job on $(hostname): $reason (exit $rc); logs in $RUN_DIR/logs" || true
  fi
  rm -f "$servers/${job}"_*.url
  if (( ${#pids[@]} )); then
    kill "${pids[@]}" 2> /dev/null || true
    wait "${pids[@]}" 2> /dev/null || true
  fi
  [[ -z $scratch ]] || rm -rf "$scratch"
}
trap on_exit EXIT
trap 'reason="received SIGTERM (cancelled or out of time)"; exit 143' TERM
fail() {
  reason=$1
  echo "$1" >&2
  exit 1
}
# The GPUs must not idle without a driver: stop when it is done or has gone silent.
check_driver() {
  local beat
  [[ ! -e $RUN_DIR/driver.done ]] || exit 0
  beat=$(stat -c %Y "$RUN_DIR/driver.alive" 2> /dev/null) || fail "the driver's heartbeat file is gone"
  (( $(date +%s) - beat <= DRIVER_STALE_S )) || fail "the driver has not heartbeated for over ${DRIVER_STALE_S}s"
}

[[ $N_SERVERS =~ ^[1-9][0-9]*$ && $GPUS_PER_SERVER =~ ^[1-9][0-9]*$ ]] ||
  fail "N_SERVERS and GPUS_PER_SERVER must be positive integers"
need=$(( N_SERVERS * GPUS_PER_SERVER ))
# Slurm's device list need not be 0..N-1, so split it rather than renumbering.
if [[ -n ${CUDA_VISIBLE_DEVICES:-} ]]; then
  IFS=, read -r -a gpus <<< "$CUDA_VISIBLE_DEVICES"
else
  read -r -a gpus <<< "$(seq -s ' ' 0 $(( ${SLURM_GPUS_ON_NODE:-$need} - 1 )))"
fi
(( need <= ${#gpus[@]} )) ||
  fail "N_SERVERS x GPUS_PER_SERVER = $need GPUs, but the job has ${#gpus[@]} (${CUDA_VISIBLE_DEVICES:-unset})"
(( need == ${#gpus[@]} )) || echo "warning: serving on $need of the job's ${#gpus[@]} GPUs" >&2
[[ -r /proc/net/tcp ]] || echo "warning: /proc/net/tcp is unreadable; ports are not checked for sharing" >&2

# --dependency=after: is also satisfied by a driver cancelled while queued.
deadline=$(( SECONDS + DRIVER_START_WAIT_S ))
until [[ -e $RUN_DIR/driver.alive || -e $RUN_DIR/driver.done ]]; do
  (( SECONDS < deadline )) || fail "the driver did not start within ${DRIVER_START_WAIT_S}s"
  sleep "$POLL_S"
done
check_driver

read -r -a ports <<< "$(free_ports "$N_SERVERS")"
(( ${#ports[@]} == N_SERVERS )) || fail "could not pick $N_SERVERS free ports"
VLLM_API_KEY=$(< "$RUN_DIR/api_key")
export VLLM_API_KEY
host=$(hostname)
scratch=$(mktemp -d "${SLURM_TMPDIR:-${TMPDIR:-/tmp}}/ttt_vllm.XXXXXX")
for ((s = 0; s < N_SERVERS; s++)); do
  mkdir -p "$scratch/$s"/{tmp,triton,vllm,xdg}
  devices=$(IFS=,; echo "${gpus[*]:s * GPUS_PER_SERVER:GPUS_PER_SERVER}")
  # Node-local compiler and kernel caches: several servers sharing a network cache directory race.
  CUDA_VISIBLE_DEVICES=$devices TMPDIR="$scratch/$s/tmp" TRITON_CACHE_DIR="$scratch/$s/triton" \
  VLLM_CACHE_ROOT="$scratch/$s/vllm" XDG_CACHE_HOME="$scratch/$s/xdg" \
    "$VLLM" serve "$MODEL_PATH" --served-model-name "$SERVED_MODEL" --host 0.0.0.0 \
    --port "${ports[s]}" --max-model-len "$MAX_MODEL_LEN" \
    --tensor-parallel-size "$GPUS_PER_SERVER" --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
    --safetensors-load-strategy lazy --enable-prefix-caching \
    > "$RUN_DIR/logs/vllm_${job}_$s.log" 2>&1 &
  pids+=($!)
  echo "server $s: GPUs $devices, port ${ports[s]}"
done

published=()
while (( ${#published[@]} < N_SERVERS )); do
  check_driver
  for ((s = 0; s < N_SERVERS; s++)); do
    [[ -z ${published[s]:-} ]] || continue
    case $(probe_server "${pids[s]}" "${ports[s]}") in
      ready)
        write_atomic "$servers/${job}_$s.url" "http://$host:${ports[s]}"
        published[s]=1
        echo "server $s ready at http://$host:${ports[s]}" ;;
      exited) fail "vLLM server $s exited during startup; see $RUN_DIR/logs/vllm_${job}_$s.log" ;;
      foreign) fail "port ${ports[s]} of server $s is also held by another process" ;;
    esac
  done
  (( ${#published[@]} == N_SERVERS )) || sleep "$POLL_S"
done
while :; do
  sleep "$POLL_S"
  check_driver
  for ((s = 0; s < N_SERVERS; s++)); do
    kill -0 "${pids[s]}" 2> /dev/null || fail "vLLM server $s exited; see $RUN_DIR/logs/vllm_${job}_$s.log"
  done
done

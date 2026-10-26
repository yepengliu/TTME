#!/usr/bin/env bash
# Submit one run on an existing evaluation pool: the CPU driver job first, then the vLLM GPU job with
# --dependency=after:<driver>, so the GPUs are never allocated while the driver is still queued.
# The driver waits (on cheap CPU) for the servers to publish their URLs and evaluates on the pool's
# endpoint directory, which it rescans, so workers that come up later join the run. The GPU job
# ends when the driver does; the driver gives up when the GPU job fails.
#
# Required: CONFIG, RUN_DIR, POOL_DIR, MODEL_PATH.
# Optional:
#   MIN_WORKERS        workers that must have written endpoints before submitting
#                      (default: half the pool, rounded up)
#   N_SERVERS          vLLM servers in the GPU job (default 4), GPUS_PER_SERVER GPUs each (default 1);
#                      the GPU job requests exactly N_SERVERS x GPUS_PER_SERVER GPUs
#   RESUME=1           continue the run in RUN_DIR from its last committed step
#   STOP_POOL_ON_EXIT=1  stop the pool's workers when the driver ends
#   DRIVER_SBATCH_ARGS, VLLM_SBATCH_ARGS  extra sbatch arguments, split on whitespace
# Paths must not contain commas: they travel through sbatch --export.
set -euo pipefail
: "${CONFIG:?}" "${RUN_DIR:?}" "${POOL_DIR:?}" "${MODEL_PATH:?}"
N_SERVERS=${N_SERVERS:-4}
GPUS_PER_SERVER=${GPUS_PER_SERVER:-1}
RESUME=${RESUME:-0}
HERE=$(cd "$(dirname "$0")" && pwd)
SBATCH=${SBATCH:-sbatch}
RUN_DIR=$(realpath -m "$RUN_DIR")
POOL_DIR=$(realpath -m "$POOL_DIR")
CONFIG=$(realpath "$CONFIG")
die() { echo "$1" >&2; exit 1; }

for value in "$CONFIG" "$RUN_DIR" "$POOL_DIR" "$MODEL_PATH" "$HERE"; do
  [[ $value != *,* ]] || die "paths must not contain commas (sbatch --export): $value"
done
[[ $N_SERVERS =~ ^[1-9][0-9]*$ && $GPUS_PER_SERVER =~ ^[1-9][0-9]*$ ]] ||
  die "N_SERVERS and GPUS_PER_SERVER must be positive integers"
[[ -d $POOL_DIR/endpoints ]] || die "$POOL_DIR is not an evaluation pool (no endpoints/)"
if [[ -z ${MIN_WORKERS:-} ]]; then
  [[ -s $POOL_DIR/size ]] || die "set MIN_WORKERS: $POOL_DIR/size is missing"
  MIN_WORKERS=$(( ($(< "$POOL_DIR/size") + 1) / 2 ))
fi
[[ $MIN_WORKERS =~ ^[1-9][0-9]*$ ]] || die "MIN_WORKERS must be a positive integer"
case $RESUME in
  0) [[ -z $(ls -A "$RUN_DIR/run" 2> /dev/null) ]] || die "$RUN_DIR/run already holds a run; set RESUME=1 to continue it" ;;
  1) [[ -n $(ls -A "$RUN_DIR/run" 2> /dev/null) ]] || die "RESUME=1, but $RUN_DIR/run holds no run" ;;
  *) die "RESUME must be 0 or 1" ;;
esac
if [[ -e $RUN_DIR/driver.alive && ! -e $RUN_DIR/driver.done ]] &&
  (( $(date +%s) - $(stat -c %Y "$RUN_DIR/driver.alive") < ${DRIVER_STALE_S:-600} )); then
  die "a driver still seems to be running on $RUN_DIR (driver.alive is fresh)"
fi

shopt -s nullglob
endpoints=("$POOL_DIR"/endpoints/*.json)
if (( ${#endpoints[@]} < MIN_WORKERS )); then
  die "only ${#endpoints[@]} of $MIN_WORKERS evaluation workers are up; try again later"
fi
umask 077
mkdir -p "$RUN_DIR/servers" "$RUN_DIR/logs"
# mkdir -m would only cover the last component; RUN_DIR (it holds the API key) may already exist.
chmod 700 "$RUN_DIR" "$RUN_DIR/servers" "$RUN_DIR/logs"
rm -f "$RUN_DIR"/servers/*.url "$RUN_DIR/servers/FAILED" "$RUN_DIR/driver.done" "$RUN_DIR/driver.alive"
[[ -s $RUN_DIR/api_key ]] || python3 -c 'import secrets; print(secrets.token_hex(24))' > "$RUN_DIR/api_key"

read -r -a driver_args <<< "${DRIVER_SBATCH_ARGS:-}"
read -r -a vllm_args <<< "${VLLM_SBATCH_ARGS:-}"
common=ALL,CONFIG="$CONFIG",RUN_DIR="$RUN_DIR",POOL_DIR="$POOL_DIR",MODEL_PATH="$MODEL_PATH",N_SERVERS="$N_SERVERS"
common+=,GPUS_PER_SERVER="$GPUS_PER_SERVER"
driver=$("$SBATCH" --parsable --export="$common",RESUME="$RESUME",STOP_POOL_ON_EXIT="${STOP_POOL_ON_EXIT:-0}" \
  --output="$RUN_DIR/logs/driver_%j.out" ${driver_args[@]+"${driver_args[@]}"} "$HERE/driver.sbatch")
driver=${driver%%;*}
if ! servers=$("$SBATCH" --parsable --dependency="after:$driver" --export="$common",TTT_SCRIPTS="$(dirname "$HERE")" \
  --gpus="$(( N_SERVERS * GPUS_PER_SERVER ))" --output="$RUN_DIR/logs/vllm_%j.out" \
  ${vllm_args[@]+"${vllm_args[@]}"} "$HERE/vllm_servers.sbatch"); then
  # The driver is already queued: tell it to give up as soon as it starts.
  echo "submit_run.sh: the vLLM job could not be submitted" > "$RUN_DIR/servers/FAILED"
  die "the vLLM job could not be submitted; driver job $driver will exit as soon as it starts"
fi
echo "driver job $driver, vLLM job ${servers%%;*} (starts once the driver runs); ${#endpoints[@]} workers up"

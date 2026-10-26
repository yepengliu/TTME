#!/usr/bin/env bash
# Submit an evaluation pool of N CPU workers as one job array; each writes POOL_DIR/endpoints/<job id>.json.
#
# Order of a cluster run, chosen so the GPUs are never allocated while they have nothing to do:
#   1. submit_pool.sh N POOL_DIR    CPU evaluation workers first: they are the slowest to schedule.
#   2. submit_run.sh ...            once enough workers have written endpoints, it submits the
#                                   CPU driver job, then the GPU job whose only work is `vllm serve`.
# `touch POOL_DIR/STOP` ends every worker of the pool. POOL_DIR holds the workers' tokens, so it is
# made private (mode 700). It must not contain commas: it travels through sbatch --export.
set -euo pipefail

usage="usage: submit_pool.sh N_WORKERS POOL_DIR [extra sbatch arguments...]"
N=${1:?$usage}
POOL_DIR=$(realpath -m "${2:?$usage}")
shift 2
HERE=$(cd "$(dirname "$0")" && pwd)
SBATCH=${SBATCH:-sbatch}

[[ $N =~ ^[1-9][0-9]*$ ]] || { echo "N_WORKERS must be a positive integer" >&2; exit 1; }
[[ $POOL_DIR != *,* ]] || { echo "POOL_DIR must not contain commas" >&2; exit 1; }
if [[ -e "$POOL_DIR/jobid" ]]; then
  echo "$POOL_DIR already holds a pool" >&2
  exit 1
fi
umask 077
mkdir -p "$POOL_DIR/endpoints" "$POOL_DIR/logs"
# mkdir -m would only cover the last component; POOL_DIR itself may already exist.
chmod 700 "$POOL_DIR" "$POOL_DIR/endpoints" "$POOL_DIR/logs"
echo "$N" > "$POOL_DIR/size"
jobid=$("$SBATCH" --parsable --array="1-$N" --export=ALL,POOL_DIR="$POOL_DIR" \
  --output="$POOL_DIR/logs/worker_%A_%a.out" "$@" "$HERE/eval_worker.sbatch")
echo "$jobid" > "$POOL_DIR/jobid"
echo "submitted $N workers as array job $jobid"

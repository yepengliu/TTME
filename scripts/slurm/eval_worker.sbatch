#!/bin/bash
#SBATCH --job-name=ttt-eval
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --qos=YOUR_QOS
#SBATCH --partition=YOUR_CPU_PARTITION
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --time=48:00:00
# One evaluation worker with one sandbox slot per core, on a port it picks itself; it writes its
# endpoint file once listening and exits when POOL_DIR/STOP appears. A worker that dies on its own
# fails the job. Size --mem for the task: an ac1 sandbox needs several GB, ac2 and erdos far less.
set -euo pipefail
: "${POOL_DIR:?set POOL_DIR (submit_pool.sh exports it)}" "${SLURM_JOB_ID:?}" "${SLURM_CPUS_PER_TASK:?}"
POLL_S=${POLL_S:-30}
PYTHON=${PYTHON:-python}
endpoint=$POOL_DIR/endpoints/$SLURM_JOB_ID.json

"$PYTHON" -m ttt worker --host 0.0.0.0 --capacity "$SLURM_CPUS_PER_TASK" --endpoint-file "$endpoint" &
worker=$!
trap 'kill "$worker" 2> /dev/null || true; rm -f "$endpoint"' EXIT
while [[ ! -e $POOL_DIR/STOP ]]; do
  if ! kill -0 "$worker" 2> /dev/null; then
    rc=0
    wait "$worker" || rc=$?
    echo "the evaluation worker exited with status $rc before POOL_DIR/STOP" >&2
    exit $(( rc == 0 ? 1 : rc ))
  fi
  sleep "$POLL_S"
done

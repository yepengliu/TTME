# Usage guide

[← Back to TTME](../README.md)

Run all commands from the repository root.

## Requirements

- Python 3.10 or newer.
- Linux for evaluation: the sandbox and the evaluation workers run only on Linux.

## Install

`-c requirements.lock` applies the versions the reported runs used.

On the machines that run the driver or evaluate programs:

```bash
pip install -c requirements.lock -e ".[programs,test]"
```

On the machines that serve the model, install the `serve` extra separately:

```bash
pip install -c requirements.lock -e ".[serve]"
```

- `programs`: packages that generated programs commonly import (scipy, cvxpy, tqdm). Install them
  wherever programs are evaluated.
- `serve`: vLLM, needed only on the machines that serve the model.
- `test`: pytest.

On a single machine that does both, install all three extras into one environment:
`pip install -c requirements.lock -e ".[programs,serve,test]"`.

## Configs

Each experiment has a YAML file in `configs/experiments/`. The file names a model
configuration in `configs/models/` (see [Models](#models)).

For each task (`ac1`, `ac2`, `erdos`):

| Experiment | Memory |
|---|---|
| `<task>_baseline` | none |
| `<task>_memory` | Guideline Memory and Strategy Memory |
| `<task>_memory_no_guidelines` | Strategy Memory |
| `<task>_memory_no_memo` | Guideline Memory |
| `<task>_memory_recipe_only` | Parent program's change history only |
| `<task>_baseline_8b`, `<task>_memory_8b` | Same settings, using Qwen3-8B |

When memory is enabled, the prompt also includes the change history of the selected
parent program. This history includes the scores before and after each change.
In experiment filenames, `memo` refers to Strategy Memory.

Each experiment runs for 50 steps. Each step uses 8 parents and generates 64 children
per parent. Each program has a 1100 s time limit and one CPU.
The configuration loader rejects unknown keys and inconsistent settings.
Configuration files do not contain model weights. Pass the path to the model weights
when you start the servers.

## Models

| Model config | Model | Revision | Prompt format |
|---|---|---|---|
| `configs/models/gpt-oss-120b.yaml` | Hugging Face `openai/gpt-oss-120b` | `b5c939de8f754692c1647ca79fbf85e8c1e70f8a` | harmony, two-phase completion |
| `configs/models/8b-chatml.yaml` | Hugging Face `Qwen/Qwen3-8B` | `b968826d9c46dd6066d109eabc6255188de91218` | ChatML with a `<think>` block, single-phase completion |

The reported runs used these model revisions with vLLM 0.22.0. This vLLM version is
specified in `requirements.lock`. Download the specified model revision. Set
`MODEL_PATH` to its local directory. If you supply only a hub ID, vLLM downloads the
latest revision.

## Run on one machine

`scripts/local/run_local.sh` starts one vLLM server on a free local port. The script
waits until the server is ready. It then runs the experiment with local sandbox
workers. When the experiment ends, the script stops the server.

```bash
export MODEL_PATH=models/gpt-oss-120b   # downloaded snapshot directory
scripts/local/run_local.sh configs/experiments/erdos_baseline.yaml runs/erdos_baseline 16
```

Arguments: `CONFIG OUT_DIR [EVAL_WORKERS] [extra ttt run arguments]`. Optional environment
variables: `SERVED_MODEL`, `MAX_MODEL_LEN` (default 32768), `PORT`, `PYTHON`, `VLLM`, `VLLM_LOG`
(default `OUT_DIR.vllm.log`). Missing parent directories of `OUT_DIR` and `VLLM_LOG` are created.
The 120B model needs a GPU with enough memory for its weights plus a 32k-token KV cache.

## Run on a Slurm cluster

The cluster scripts in `scripts/slurm/` run three kinds of jobs: CPU evaluation workers, a CPU
driver, and a GPU job whose only work is serving the model. First replace the `#SBATCH`
placeholders (`YOUR_ACCOUNT`, `YOUR_QOS`, `YOUR_CPU_PARTITION`, `YOUR_GPU_PARTITION`) in the
`.sbatch` files, or pass the same options as extra sbatch arguments.

1. Start an evaluation pool of N workers. Each worker serves one sandbox slot per CPU and writes its
   endpoint file to `POOL_DIR/endpoints/`.

   ```bash
   scripts/slurm/submit_pool.sh 32 pools/eval --mem=160G --time=48:00:00
   ```

   An `ac1` program can need several GB, so size `--mem` for 16 slots accordingly; `ac2` and
   `erdos` need far less.

2. Once enough workers are up (default: half the pool), submit the run. The driver job is submitted
   first; the GPU job starts only after the driver is running.

   ```bash
   CONFIG=configs/experiments/ac2_memory.yaml \
   RUN_DIR=runs/ac2_memory POOL_DIR=pools/eval MODEL_PATH=models/gpt-oss-120b \
   N_SERVERS=4 STOP_POOL_ON_EXIT=1 \
   scripts/slurm/submit_run.sh
   ```

   | Variable | Meaning |
   |---|---|
   | `N_SERVERS`, `GPUS_PER_SERVER` | vLLM servers in the GPU job (default 4) and GPUs per server (default 1); the job requests exactly their product |
   | `MIN_WORKERS` | workers that must be up before submitting (default: half the pool) |
   | `STOP_POOL_ON_EXIT=1` | stop the pool when the driver ends |
   | `RESUME=1` | continue the run in `RUN_DIR` from its last committed step |
   | `DRIVER_SBATCH_ARGS`, `VLLM_SBATCH_ARGS` | extra sbatch arguments for the two jobs |
   | `SERVED_MODEL`, `MAX_MODEL_LEN`, `GPU_MEMORY_UTILIZATION`, `PYTHON`, `VLLM` | passed through to the jobs |

   The run's output is in `RUN_DIR/run`; job logs are in `RUN_DIR/logs`.

Workers that start later join the run automatically. When the driver finishes, the GPU job stops.
If the GPU job fails, the driver stops. Resubmit the run with `RESUME=1`.
Run `touch POOL_DIR/STOP` to stop all workers in the pool.
Paths must not contain commas because `sbatch --export` uses commas to separate values.

## Run `ttt run` directly

With servers you started yourself:

```bash
ttt run --config configs/experiments/ac1_memory.yaml \
  --vllm-url http://host1:8000 --vllm-url http://host2:8000 --served-model my-model \
  --out-dir runs/ac1_memory --local-eval 32
```

- `--local-eval N` evaluates on this machine with N sandbox workers. `--eval-workers PATH` uses a
  pool instead: a directory of worker endpoint files, or a file listing them. It is rescanned
  during the run.
- `--resume` continues from the last committed step.
- `--api-key-env VAR` sends the bearer key held in environment variable `VAR`.
- `--memo-url URL` sends memory-related requests to a different server.
- `--no-save-code` skips saving every rollout's program.

`ttt worker --capacity N --endpoint-file FILE [--host 0.0.0.0] [--port 0]` runs one evaluation
worker by hand. `python -m ttt ...` works the same as `ttt ...`.

## Output

| Path in the run directory | Content |
|---|---|
| `manifest.json` | resolved config and its hash |
| `steps.jsonl` | one line per step: `best` (best archived score), `best_valid` (best valid program so far), `valid`, `failed`, `infra` (infrastructure failures), `seconds` |
| `rollouts.jsonl` | one line per rollout: parent and child ids and scores, validity, archive admission, error, program summary, captured output, strategy assignment |
| `events.jsonl` | structured log |
| `checkpoints/step_NNNNNN/` | archive, memory and memo state after each committed step |
| `constructions/`, `codes/` | saved constructions and programs, named by their content hashes |
| `memo/` | Strategy Memory snapshots and update log |

AC1 and Erdős are minimized and AC2 is maximized. Scores are always recomputed from the
construction a program returns.

## Score a construction

All nine TTME constructions in Tables 1 and 2 are stored with their paper scores and
SHA-256 hashes in [results/manifest.json](../results/manifest.json). Verify the full set with
CPU and NumPy only; this does not run a model or a generated optimizer:

```bash
python scripts/verify_results.py
```

To score one of the TTT-Discover constructions:

```bash
ttt score erdos results/erdos/sequence.json
```

The JSON file holds either a list of floats or `{"sequence": [...]}`.

## Tests

```bash
pytest
```

## Safety

Evaluation runs model-written Python with the permissions of the account that runs the workers.
The sandbox limits execution time and output size. It also cleans up processes.
It does not isolate programs from the filesystem or the network.
Run workers in a container or under a separate account.

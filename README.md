# TTME

Code and constructions for **Can Test-Time Memory Evolution Outperform Test-Time Training for Scientific Discovery?**

[Paper](paper.pdf) · [Reported constructions](results/README.md)

Test-time search with memory for mathematical constructions. A served LLM writes programs for a
construction task, each program is run and scored in a sandbox, a PUCT archive selects the next
parents, and an optional memory layer (lineage recipe, Guideline Memory, method memos) adds
context to every prompt.

This release contains the TTT-Discover search implementation for AC1, AC2 and Erdős,
with the V2 control card selected in every memo-enabled experiment. The
[results directory](results/README.md) also contains the three ThetaEvolve constructions
reported in Table 2 and a standalone scoring command. The ThetaEvolve search runner is
not included in this release.

Run all commands below from the repository root.

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

An experiment is one YAML file in `configs/experiments/`; the model config it names is in
`configs/models/` (see [Models](#models)).

For each task (`ac1`, `ac2`, `erdos`):

| Experiment | Memory |
|---|---|
| `<task>_baseline` | none |
| `<task>_memory` | lineage recipe, Guideline Memory and method memos |
| `<task>_memory_no_guidelines` | lineage recipe and method memos |
| `<task>_memory_no_memo` | lineage recipe and Guideline Memory |
| `<task>_memory_recipe_only` | lineage recipe |
| `<task>_baseline_8b`, `<task>_memory_8b` | as above, with the `8b-chatml` model |

Every experiment runs 50 steps of 8 parents × 64 children, with a 1100 s limit and one CPU per
program. Unknown keys and inconsistent settings are rejected when a config is loaded. The model
weights are not part of a config: you pass the path when you start the servers.

## Models

| Model config | Model | Revision | Prompt format |
|---|---|---|---|
| `configs/models/gpt-oss-120b.yaml` | Hugging Face `openai/gpt-oss-120b` | `b5c939de8f754692c1647ca79fbf85e8c1e70f8a` | harmony, two-phase completion |
| `configs/models/8b-chatml.yaml` | Hugging Face `Qwen/Qwen3-8B` | `b968826d9c46dd6066d109eabc6255188de91218` | ChatML with a `<think>` block, single-phase completion |

The reported runs used these snapshots at these revisions, served with vLLM 0.22.0 (the version
pinned in `requirements.lock`). Download the revision and pass its local directory as `MODEL_PATH`;
a bare hub id makes vLLM fetch the latest revision instead.

## Run on one machine

`scripts/local/run_local.sh` starts one vLLM server on a free local port, waits until it is
healthy, runs the experiment with local sandbox workers and stops the server at the end.

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
If the GPU job fails, the driver gives up; resubmit with `RESUME=1`. `touch POOL_DIR/STOP` ends
every worker of a pool. Paths must not contain commas, because they are passed through
`sbatch --export`.

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
- `--memo-url URL` sends memory and memo calls to a different server.
- `--no-save-code` skips saving every rollout's program.

`ttt worker --capacity N --endpoint-file FILE [--host 0.0.0.0] [--port 0]` runs one evaluation
worker by hand. `python -m ttt ...` works the same as `ttt ...`.

## Output

| Path in the run directory | Content |
|---|---|
| `manifest.json` | resolved config and its hash |
| `steps.jsonl` | one line per step: `best` (best archived score), `best_valid` (best valid program so far), `valid`, `failed`, `infra` (infrastructure failures), `seconds` |
| `rollouts.jsonl` | one line per rollout: parent and child ids and scores, validity, archive admission, error, docstring, stdout tail, memo arm |
| `events.jsonl` | structured log |
| `checkpoints/step_NNNNNN/` | archive, memory and memo state after each committed step |
| `constructions/`, `codes/` | content-addressed constructions and programs |
| `memo/` | method-library snapshots and the merge log |

AC1 and Erdős are minimized and AC2 is maximized. Scores are always recomputed from the
construction a program returns.

## Score a construction

All nine TTME constructions in Tables 1 and 2 are stored with their paper scores and
SHA-256 hashes in [results/manifest.json](results/manifest.json). Verify the full set with
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
The sandbox limits time and output size and cleans up processes, but it does not isolate
programs from the filesystem or the network. Run workers in a container or under a separate
account.

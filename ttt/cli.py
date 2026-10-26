"""Command line: `ttt run`, `ttt worker`, `ttt score`."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional, Sequence

from ttt import log, records
from ttt.config import Runtime, load_experiment

READY_TIMEOUT_S = 2400.0


def _parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="ttt", description=__doc__)
    sub = ap.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="run or resume one experiment")
    run.add_argument("--config", required=True, help="experiment YAML")
    run.add_argument("--vllm-url", action="append", required=True,
                     help="generation server base URL; repeat for several servers")
    run.add_argument("--served-model", required=True, help="model name the servers serve")
    run.add_argument("--out-dir", required=True)
    evals = run.add_mutually_exclusive_group(required=True)
    evals.add_argument("--eval-workers",
                       help="endpoints directory or list file; rescanned during the run")
    evals.add_argument("--local-eval", type=int, help="number of local sandbox workers")
    run.add_argument("--resume", action="store_true", help="continue from the last committed step")
    run.add_argument("--gen-concurrency", type=int, default=64)
    run.add_argument("--eval-concurrency", type=int, default=None,
                     help="concurrent evaluations (default: evaluator capacity)")
    run.add_argument("--api-key-env", default=None,
                     help="environment variable holding the servers' bearer API key")
    run.add_argument("--memo-url", default=None,
                     help="server for memory and memo calls (default: the first --vllm-url)")
    run.add_argument("--no-save-code", dest="save_code", action="store_false",
                     help="do not keep every rollout's program under <out-dir>/codes")
    sub.add_parser("worker", help="run one evaluation worker (arguments pass through)")
    score = sub.add_parser("score", help="score a construction")
    score.add_argument("task")
    score.add_argument("sequence", help="JSON file: a list of floats or {\"sequence\": [...]}")
    return ap


def _runtime(args: argparse.Namespace) -> Runtime:
    api_key = None
    if args.api_key_env:
        api_key = os.environ.get(args.api_key_env)
        if not api_key:
            raise SystemExit(f"environment variable {args.api_key_env} is not set")
    for flag, value in (("--local-eval", args.local_eval), ("--gen-concurrency", args.gen_concurrency),
                        ("--eval-concurrency", args.eval_concurrency)):
        if value is not None and value < 1:
            raise SystemExit(f"{flag} must be >= 1")
    return Runtime(out_dir=args.out_dir, vllm_urls=tuple(args.vllm_url),
                   served_model=args.served_model, api_key=api_key,
                   eval_workers_file=args.eval_workers, local_eval_workers=args.local_eval or 0,
                   gen_concurrency=args.gen_concurrency, eval_concurrency=args.eval_concurrency,
                   resume=args.resume, memo_url=args.memo_url, save_code=args.save_code)


def _wait_ready(clients: list) -> None:
    for client in clients:
        log.event("waiting_for_server", url=client.base_url)
        client.wait_ready(READY_TIMEOUT_S)
        # /health can answer before the engine serves completions; one token proves it does.
        client.generate("ready?", max_tokens=1, temperature=0.0)
        log.event("server_ready", url=client.base_url)


def _evaluator(runtime: Runtime):
    if runtime.eval_workers_file:
        from ttt.eval.remote import RemoteEvaluator
        return RemoteEvaluator(runtime.eval_workers_file)
    from ttt.eval.local import LocalEvaluator
    return LocalEvaluator(runtime.local_eval_workers)


def cmd_run(args: argparse.Namespace) -> int:
    """Build every component from the config and command line, then run the loop."""
    from ttt import loop, tasks
    from ttt.llm import VLLMClient, build_format

    exp = load_experiment(args.config)
    runtime = _runtime(args)
    records.claim_out_dir(runtime.out_dir, runtime.resume)
    log.configure(runtime.out_dir)
    log.event("start", experiment=exp.name, resume=runtime.resume,
              runtime=records.runtime_record(runtime))
    task = tasks.get(exp.task)
    fmt = build_format(exp.model, exp.sampling)

    def client(url: str) -> VLLMClient:
        return VLLMClient(url, runtime.served_model, api_key=runtime.api_key,
                          timeout_s=runtime.request_timeout_s)

    clients = [client(u) for u in runtime.vllm_urls]
    aux_client = client(runtime.memo_url) if runtime.memo_url else clients[0]
    _wait_ready(clients + ([aux_client] if runtime.memo_url else []))
    memory = memo = None
    if exp.memory is not None:
        from ttt.memory import build_memory
        memory = build_memory(exp.memory, task=task, fmt=fmt, client=aux_client, aux=exp.model.aux)
    if exp.memo is not None:
        from ttt.memo import build_memo
        memo = build_memo(exp.memo, task=task, fmt=fmt, client=aux_client, aux=exp.model.aux,
                          children_per_parent=exp.search.children_per_parent,
                          context_window=exp.sampling.context_window, out_dir=runtime.out_dir)
    result = loop.run(exp, runtime, task=task, evaluator=_evaluator(runtime), clients=clients,
                      fmt=fmt, memory=memory, memo=memo)
    print(json.dumps({"experiment": exp.name, "best": result["best"],
                      "best_valid": result["best_valid"], "steps": len(result["steps"]),
                      "out_dir": runtime.out_dir}))
    return 0


def cmd_score(args: argparse.Namespace) -> int:
    """Print the task score of a construction stored as JSON."""
    from ttt import tasks

    data = json.loads(Path(args.sequence).read_text(encoding="utf-8"))
    seq = data["sequence"] if isinstance(data, dict) else data
    print(repr(float(tasks.score_sequence(args.task, seq))))
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Entry point of the `ttt` command."""
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "worker":
        from ttt.eval import worker
        return worker.main(argv[1:]) or 0
    args = _parser().parse_args(argv)
    if args.command == "run":
        return cmd_run(args)
    if args.command == "score":
        return cmd_score(args)
    _parser().error(f"unknown command {args.command}")
    return 2

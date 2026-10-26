"""One boundary: select transitions, mine notes, merge them, rewrite family memories."""
from __future__ import annotations

import json
import os
from dataclasses import asdict

from ttt.log import event, warn
from ttt.memo.family import distill
from ttt.memo.mining import expand_notes, mine
from ttt.memo.rounds import (MergeResult, MergeSettings, merge_rounds, roll_failed, status_of,
                             update_health)
from ttt.memo.transitions import prune, select

_SUMMED = ("new_notes", "absorbed", "solo_new", "declared_closest", "relabel_merged",
           "redirected")


def merge_settings(loop) -> MergeSettings:
    """Merge limits of a MemoLoop's config."""
    b = loop.cfg.budgets
    return MergeSettings(ctx=loop.context_window, margin=b.merge_ctx_margin,
                         start_max_tokens=loop.aux.merge_max_tokens,
                         max_tokens_cap=b.merge_max_tokens_cap, drain=loop.cfg.merge_drain,
                         drain_min_notes=b.merge_drain_min_notes, max_rounds=b.merge_max_rounds,
                         rounds_budget_s=b.merge_rounds_budget_s, call_timeout_s=b.call_timeout_s)


def consolidate(rebuilds: list, start_family: dict, families: dict) -> dict:
    """One rebuild record for all successful rounds of a boundary."""
    first, last = rebuilds[0], rebuilds[-1]
    partition = {}
    for fid, fam in families.items():
        src = sorted({start_family[c] for c in fam.variants
                      if c in start_family and start_family[c] != fid})
        if src:
            partition[fid] = src
    out = {"families_in": first["families_in"], "families_out": last["families_out"],
           "variants_in": first["variants_in"], "variants_out": last["variants_out"],
           "reused": last["reused"], "rounds": len(rebuilds), "fused": len(partition),
           "partition": partition}
    out.update({k: sum(int(r.get(k) or 0) for r in rebuilds) for k in _SUMMED})
    return out


def _summary(res: MergeResult, pending_in: int, families_in: int, families_out: int) -> dict:
    ok = [r for r in res.rounds if r.get("outcome") == "ok"]
    out = {"status": status_of(res), "stop": res.stop, "rounds_ok": len(ok),
           "rounds_attempted": len(res.rounds), "margin": res.margin,
           "families_in": families_in, "families_out": families_out, "pending_in": pending_in,
           "pending_out": len(res.queue), "deferred": res.deferred,
           "notes_shown_ok": res.shown_ok, "rounds": res.rounds,
           "len_fail": sum(1 for r in res.rounds if r.get("outcome") == "finish_length"),
           "call_fail": sum(1 for r in res.rounds if r.get("outcome") == "call_error")}
    for k in ("placed", "attached", "notes_new", "created"):
        out[k] = sum(int(r.get(k) or 0) for r in ok)
    if not ok:
        out["reason"] = [r.get("outcome") for r in res.rounds]
    return out


def run_boundary(loop, step: int, archive_ids: frozenset) -> None:
    """Run the boundary of `step` on a MemoLoop's state; model failures degrade, never raise."""
    cfg = loop.cfg
    picks, diag = select(loop.pool, loop.codes, loop.seen, archive_ids,
                         want=cfg.notes_per_boundary, step=step, reserve=cfg.stepbest_reserve,
                         maximize=loop.task.maximize, window=loop.refused_window)
    rec = {"step": step, "sampler": diag}
    if not picks:
        return _finish(loop, step, archive_ids,
                       dict(rec, status="skipped", reason="nothing_fresh"), [])
    mined = mine(loop.llm, picks, loop.codes, loop.window, loop.aux, cfg.budgets)
    rec["mining"] = {"picked": len(picks), "memos": len(mined)}
    if not mined:
        warn("memo_mining_failed", step=step, picked=len(picks))
        return _finish(loop, step, archive_ids,
                       dict(rec, status="failed", reason="zero_memos"), [])
    notes, rec["n_split"] = expand_notes(mined, step)
    loop.seen.update(t.sig for t, *_rest in mined)
    start_family = {cid: v.family for cid, v in loop.library.variants.items()}
    families_in, pending_in = len(loop.library.families), len(loop.pending)
    res = merge_rounds(loop.llm, loop.library, loop.pending + notes, step,
                       merge_settings(loop), loop.failed_hashes)
    loop.pending = res.queue
    if res.rounds:
        loop.failed_hashes = roll_failed(loop.failed_hashes, res.new_failed)
    rec.update(_summary(res, pending_in + len(notes), families_in, len(loop.library.families)))
    if res.rebuilds:
        rec["rebuild"] = consolidate(res.rebuilds, start_family, loop.library.families)
        budget = cfg.budgets.family_budget_s
        if len(res.rounds) > 1:
            budget = max(budget, cfg.budgets.merge_distill_reserve_s)
        rec["family_memory"], loop.carry = distill(
            loop.llm, loop.library, step, loop.carry, budget_s=budget,
            workers=cfg.budgets.family_workers, max_tokens=loop.aux.family_max_tokens)
        rec["family_memory_budget_s"] = budget
    last_rung = next((r.get("rung") for r in reversed(res.rounds) if r.get("outcome") == "ok"),
                     res.rounds[-1].get("rung") if res.rounds else None)
    _finish(loop, step, archive_ids, rec, res.rounds, last_rung)


def _finish(loop, step: int, archive_ids: frozenset, rec: dict, rounds: list,
            last_rung=None) -> None:
    loop.health = update_health(loop.health, step, rec["status"], rounds,
                                len(loop.library.families), loop.pending, last_rung)
    if loop.health["frozen"]:
        warn("memo_merge_frozen", step=step, pending=len(loop.pending),
             failed_boundaries=loop.health["consecutive_failed_boundaries"])
    loop.pool, loop.codes = prune(loop.pool, loop.codes, loop.seen, archive_ids,
                                  loop.refused_window)
    loop.last_boundary = step
    rec["merge_health"] = loop.health
    event("memo_boundary", step=step, status=rec["status"], families=len(loop.library.families),
          pending=len(loop.pending), stop=rec.get("stop"), placed=rec.get("placed"),
          rounds=rec.get("rounds_attempted", 0))
    _write_files(loop, step, rec)


def log_boundary(path: str, step: int, rec: dict) -> None:
    """Append `rec` to the merge log after dropping records of boundary `step` or later."""
    kept, dropped = [], 0
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    old = json.loads(line)
                except json.JSONDecodeError:
                    warn("memo_merge_log_bad_line", path=path, line=line[:120])
                    continue
                at = old.get("step") if isinstance(old, dict) else None
                if isinstance(at, int) and at < step:
                    kept.append(line.rstrip("\n") + "\n")
                else:
                    dropped += 1
    if dropped:
        # A resumed run repeats this boundary; its earlier records describe a dropped branch.
        event("memo_merge_log_trimmed", step=step, dropped=dropped, kept=len(kept))
    with open(path + ".tmp", "w", encoding="utf-8") as f:
        f.writelines(kept + [json.dumps(rec, default=str) + "\n"])
    os.replace(path + ".tmp", path)


def _write_files(loop, step: int, rec: dict) -> None:
    if not loop.out_dir:
        return
    d = os.path.join(loop.out_dir, "memo")
    snap = {"step": step, "library": loop.library.to_dict(),
            "pending": [asdict(n) for n in loop.pending], "merge_health": loop.health}
    path = os.path.join(d, "library_step_%06d.json" % step)
    try:
        os.makedirs(d, exist_ok=True)
        log_boundary(os.path.join(d, "merge_log.jsonl"), step, rec)
        with open(path + ".tmp", "w", encoding="utf-8") as f:
            json.dump(snap, f, indent=1, default=str)
        os.replace(path + ".tmp", path)
    except OSError as exc:
        warn("memo_snapshot_failed", step=step, path=path, error=str(exc)[:300])

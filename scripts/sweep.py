"""Multi-seed local architecture sweep for FactWorld (natural-language format).

For each (task x architecture x seed): train from scratch and score the canonical
**match** metric on every held-out eval length, plus a **prefix-match decomposition**
that separates the legs of multi-token answers (e.g. composite = holder leg then
value leg). Composition is bimodal, so a cell is reported as its per-seed values —
a mean alone turns "one seed formed, four floored" into an indistinguishable
mid-range number.

Every table carries the shallow-adversary FLOOR rows for the cell, recomputed from
the exact deterministic items that were scored (``factworld.validity``). A score is
only readable against the LARGEST of them: on s5_chain the initial-map chase (ignore
the events, dereference the stated initial map) is worth 0.16-0.38 depending on
(k, depth, L), and on the low-k cells the plain "never echo the queried agent"
guesser at 1/(k-1) is worth more.

Per-run records carry the training loss curve, the held-out loss at each eval
length, and the model's predictions, so a floored cell can be diagnosed after the
fact (undertrained vs formed-but-mis-scored vs at the adversary floor) without
re-running it.

Crash-safe: each completed run is appended to a JSONL log as it finishes, so
partial results survive an interrupt. Predictions go to a `<prefix>.preds.jsonl`
sidecar (example index, gold, prediction — the prompt is recomputable from
(spec, split, length, idx)). A markdown + JSON summary is (re)written at the end
and after every aggregation pass.

Example:
    .venv-train/bin/python scripts/sweep.py \\
        --tasks binding_v2,composite_copy_v2 \\
        --archs gdp_hybrid,fprm,transformer \\
        --seeds 0 1 2 3 4 --steps 8000 --d_model 256
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from datetime import datetime, timezone

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from factworld import tasks as TK
from factworld.backends import LocalBackend
from factworld.render import Renderer
from factworld.runner import evaluate_task
from factworld.validity import S5_CHAIN_ADVERSARIES, operative_floor, s5_chain_floors

# ``factworld.train`` imports torch at module scope; it is pulled in inside run_one so the
# reporting layer (floors, aggregation, markdown) stays importable — and testable — without a
# GPU environment.


def build_docs(examples, use_trace=False, interleaved=False):
    """prompt + (optional oracle worked-trace) + answer, single-space separated.

    ``interleaved`` uses meta["interleaved_prompt"] — checkpoint tokens inside the
    event stream (the protocol that formed s5) instead of a trace appended after
    the query; evaluation stays free-running on the plain prompt."""
    docs = []
    for e in examples:
        if interleaved and "interleaved_prompt" in e.meta:
            docs.append(f"{e.meta['interleaved_prompt']} {e.answer}")
        elif use_trace and "trace" in e.meta:
            docs.append(f"{e.prompt} {e.meta['trace']} {e.answer}")
        else:
            docs.append(f"{e.prompt} {e.answer}")
    return docs


def _content_tokens(s):
    """Normalized tokens with punctuation stripped: the semantic answer span."""
    return [t for t in Renderer.normalize(s).split() if t != "."]


def prefix_decomp(inspected, trace_mode=False):
    """Prefix-match decomposition over (prompt, gold, pred, correct) tuples, on CONTENT tokens.

    For a 2-token composite answer (holder, value) this yields prefix {0: neither, 1: holder-only,
    2: both} -- a direct read of where composition breaks. `holder_acc` is first-content-token
    accuracy (the binding/state leg); `value_acc` is second-content-token accuracy over the
    2-content-token answers (the recall leg of composition).

    In ``trace_mode`` the prediction is a self-generated scratchpad (trace) FOLLOWED BY the
    answer, so we score the LAST len(gold) content tokens (the committed answer), not the
    prefix -- otherwise the trace tokens are misread as the answer.
    """
    n = len(inspected)
    buckets = {0: 0, 1: 0, 2: 0}
    leg1 = 0           # first content token correct (holder / single-token answer)
    two_token = 0      # answers with >=2 content tokens (composite)
    leg2 = 0           # second content token correct (value), over two_token answers
    for _prompt, gold, pred, _ok in inspected:
        g = _content_tokens(gold)
        p = _content_tokens(pred)
        if trace_mode and len(p) >= len(g):
            p = p[-len(g):]                          # score the committed answer (tail), not the trace
        k = 0
        while k < len(g) and k < len(p) and p[k] == g[k]:
            k += 1
        buckets[min(k, 2)] = buckets.get(min(k, 2), 0) + 1
        if len(p) >= 1 and len(g) >= 1 and p[0] == g[0]:
            leg1 += 1
        if len(g) >= 2:
            two_token += 1
            if len(p) >= 2 and p[1] == g[1]:
                leg2 += 1
    return {
        "prefix": {str(k): buckets.get(k, 0) / n for k in (0, 1, 2)},
        "holder_acc": leg1 / n,
        # None (not 0.0) where no answer has a second content token: a single-token family has no
        # value leg, and printing 0.00 for it reads as a failed recall leg that was never scored.
        "value_acc": (leg2 / two_token) if two_token else None,
        "answer_tokens": max((len(_content_tokens(g)) for _p, g, _pr, _ok in inspected), default=0),
    }


def cell_floors(spec, length, n):
    """Shallow-adversary floors for the exact items this cell scores.

    Recomputed from ``generate(spec, "test", n=n, length=length)`` — the same deterministic
    call ``evaluate_task`` makes — so the floor row belongs to the cell rather than being a
    global constant. Returns {} for families with no registered adversary.

    The ``chain`` family has no event stream: nothing moves the stated map, so the
    initial-map chase reproduces the oracle and scores 1.000. It is dropped there — printing
    it as a floor would make the curriculum's stage-1 dereference arm, whose whole job is to
    reach 1.000, look like it never left its floor.
    """
    if spec.family not in ("s5_chain", "chain"):
        return {}
    return s5_chain_floors(TK.generate(spec, "test", n=n, length=length), spec.k,
                           has_events=(spec.family == "s5_chain"))


def held_out_loss(model, tok, texts, device, batch=16, max_len=1280):
    """Mean next-token cross-entropy (nats/token) over held-out documents.

    The training record's ``final_loss`` is one batch of TRAIN loss and says nothing about
    whether the run had converged or had even reached its data's entropy floor. Held-out loss
    at each eval length is the cheap scalar that separates "undertrained" from "trained and
    still at the adversary floor" — the two readings a floored cell is otherwise ambiguous
    between.
    """
    import torch
    import torch.nn.functional as F

    docs = [tok.encode(t, add_eos=True)[:max_len] for t in texts]
    docs.sort(key=len)
    model.eval()
    total, count = 0.0, 0
    with torch.no_grad():
        for i in range(0, len(docs), batch):
            chunk = docs[i:i + batch]
            ml = max(len(s) for s in chunk)
            inp = torch.full((len(chunk), ml), tok.pad_id, dtype=torch.long, device=device)
            for r, s in enumerate(chunk):
                inp[r, : len(s)] = torch.tensor(s, device=device)
            with torch.autocast(device, dtype=torch.bfloat16):
                logits = model(inp[:, :-1])
            tgt = inp[:, 1:]
            ce = F.cross_entropy(logits.float().reshape(-1, tok.vocab_size), tgt.reshape(-1),
                                 reduction="none")
            mask = (tgt != tok.pad_id).float().reshape(-1)
            total += float((ce * mask).sum())
            count += int(mask.sum())
    model.train()
    return total / max(1, count)


def _interleaved_slots(plain_prompt: str, interleaved_prompt: str):
    """Token indices of the checkpoint slots inside ``interleaved_prompt``.

    The interleaved prompt is the plain prompt with exactly one checkpoint token inserted
    after each event, so a two-pointer alignment recovers the slot positions. The insertion
    is unambiguous: the token following an event in the plain prompt is always a step label
    (``s3``) or ``what``, and a checkpoint is always a content id, so an inserted token never
    coincides with the token it precedes.

    Returns (tokens, slot_indices) where ``tokens`` is the interleaved token list.
    """
    plain = plain_prompt.split()
    inter = interleaved_prompt.split()
    slots, i = [], 0
    for j, tk in enumerate(inter):
        if i < len(plain) and tk == plain[i]:
            i += 1
        else:
            slots.append(j)
    if i != len(plain) or len(inter) - len(plain) != len(slots):
        raise RuntimeError("interleaved prompt does not align with the plain prompt")
    return inter, slots


def guided_free_run_eval(model, tok, spec, length, n, device, max_answer_tokens=6):
    """Guided free-run eval: events teacher-forced, checkpoints and answer GENERATED.

    This is the s5 formation protocol (scripts/experiment_dense_supervision.py, ``e2e_eval``)
    adapted to single-token checkpoint slots: the model runs its own state through the stream
    instead of copying a trace, but it is never asked to survive a format shift between
    training and eval.

    Free-running eval on the plain checkpoint-free prompt is not the same experiment. Under
    interleaved supervision the model has only ever seen an event followed by a checkpoint;
    the plain prompt deletes every checkpoint, so the eval prompt is off the training
    distribution before the first hop is taken. At depth 1 it is worse than that: the answer
    is a verbatim copy of the last checkpoint, i.e. of the token immediately preceding
    ``what``, which the plain prompt removes — so the cell scores a copy rule against a
    prompt with the source deleted, and measures nothing about state tracking.

    TWO DIFFERENCES FROM THE REFERENCE (experiment_dense_supervision.py:114-150). There a slot
    is a span inside a sentence, so e2e_eval generates up to 4 tokens per slot, credits the
    FIRST one whose type is valid (agent for a holder slot, value for a value slot), and
    resyncs to the next ``.``. Here a slot is exactly one token wide — the interleaved
    document places a bare id between an event and the next step label — so this reads one
    unconstrained argmax and takes it as the checkpoint, with no type filter and no resync.

    That makes ``checkpoint_acc`` a LOWER BOUND on tracking: on-format it measures the same
    quantity, and off-format (the model emits punctuation or a step label where training put
    an id) it charges an error the reference would have skipped past. The generated
    checkpoints are returned per item, so a cell that lands between its floor and the
    decision threshold can be re-scored under the reference's constrained rule from the saved
    predictions instead of being re-run.

    Returns (overall, checkpoint_acc, records) with
    records = (idx, gold, pred, correct, gen_checkpoints, gold_checkpoints).
    """
    import torch

    examples = TK.generate(spec, "test", n=n, length=length)
    eos = tok.eos_id
    hits = ck_hits = ck_total = 0
    records = []
    model.eval()
    with torch.no_grad():
        for idx, ex in enumerate(examples):
            inter = ex.meta.get("interleaved_prompt")
            if inter is None:
                raise ValueError(
                    f"{spec.name}: guided free-run eval needs meta['interleaved_prompt'] "
                    f"(spec.start_trace=True)")
            toks, slots = _interleaved_slots(ex.prompt, inter)
            gold_ck = [toks[s] for s in slots]
            slotset = set(slots)
            ids, gen_ck = [], []
            for j, tk_str in enumerate(toks):
                if j in slotset:
                    with torch.autocast(device, dtype=torch.bfloat16):
                        nx = int(model(torch.tensor([ids], device=device))[0, -1].float().argmax())
                    ids.append(nx)                                  # feed the model's OWN checkpoint
                    gen_ck.append(tok.id_to_token.get(nx, "<unk>"))
                else:
                    ids += tok.encode(tk_str)
            out = []
            for _ in range(max_answer_tokens):
                with torch.autocast(device, dtype=torch.bfloat16):
                    nx = int(model(torch.tensor([ids], device=device))[0, -1].float().argmax())
                if nx == eos:
                    break
                ids.append(nx)
                out.append(nx)
            pred = tok.decode(out)
            ok = bool(TK.score_relaxed(Renderer.normalize(pred), Renderer.normalize(ex.answer)))
            hits += ok
            ck_hits += sum(1 for a, b in zip(gen_ck, gold_ck) if a == b)
            ck_total += len(gold_ck)
            records.append((idx, ex.answer, pred, ok, gen_ck, gold_ck))
    model.train()
    return hits / max(1, len(examples)), ck_hits / max(1, ck_total), records


def _rec(i, gold, pred, ok, gen_ck, gold_ck, **extra):
    """One prediction record. Generated checkpoints ride along on the guided arm so a
    below-threshold checkpoint_acc can be re-scored under the reference's type-constrained
    rule without re-running the cell (see ``guided_free_run_eval``)."""
    d = {"idx": i, "gold": gold, "pred": pred, "correct": ok, **extra}
    if gen_ck is not None:
        d["checkpoints"] = {"gen": gen_ck, "gold": gold_ck}
    return d


def trace_budget(spec, length, n_trace):
    """Generation budget for a trace-mode arm: the oracle trace, plus honest headroom.

    The budget has to cover a scratchpad the model produces itself, so it must tolerate the
    model emitting MORE than the oracle does. ``n_trace + 6`` does not: the answer is one
    token, so it leaves five tokens of slack, and a single extra checkpoint row is k tokens
    wide — at k >= 6 one spurious row pushes a correct answer past the budget and the cell
    scores 0 for a formatting overshoot rather than a wrong answer. Headroom here is one full
    extra checkpoint row plus the answer and <eos>.

    The interaction to keep in mind: generation is greedy and token-by-token, so the budget is
    also the per-item cost. It scales with L*k on the event_trace arms (L=8, k=8 is 64 trace
    tokens before headroom), which is why eval_n and the budget have to be chosen together.
    """
    return n_trace + 2 * spec.k + 8 + max(4, length)


def run_one(spec, arch, seed, *, d_model, n_layers, steps, batch, train_n, eval_n,
            use_short_conv, use_trace, device, interleaved=False, guided_eval=False,
            loss_log_interval=100, keep_preds=10):
    """Train one config; return per-length scores, floors, diagnostics and predictions."""
    import torch

    from factworld import train as T

    d_ff = 4 * d_model
    w, r = TK.build_world(spec)
    train = TK.generate(spec, "train", n=train_n)
    tok, docs, _ = T.prepare(build_docs(train, use_trace, interleaved), [], [w], renderer=r)
    run = T.run(
        arch, tok, docs, [], steps=steps, batch=batch, d_model=d_model, n_layers=n_layers,
        d_ff=d_ff, seed=seed, return_model=True, device=device, use_short_conv=use_short_conv,
        loss_log_interval=loss_log_interval,
    )
    model = run["model"]
    backend = LocalBackend([w], arch=arch, model=model, tokenizer=tok, device=device)
    # In trace mode the model emits the full scratchpad THEN the answer, so the generation
    # budget has to cover trace + answer and the committed TAIL is what gets scored.
    # Interleaved supervision lives only in the TRAINING docs, so its eval generates no trace.
    emits_trace = use_trace and not interleaved
    out, preds = {}, []
    for L in spec.eval_lengths:
        cell = {}
        if guided_eval:
            overall, ck_acc, records = guided_free_run_eval(model, tok, spec, L, eval_n, device)
            cell["checkpoint_acc"] = ck_acc
            cell.update({"prefix": None, "holder_acc": None, "value_acc": None,
                         "answer_tokens": max((len(_content_tokens(rec[1])) for rec in records),
                                              default=0)})
        else:
            if emits_trace:
                probe = TK.generate(spec, "test", n=1, length=L)[0]
                max_new = trace_budget(spec, L, len(probe.meta.get("trace", "").split()))
            else:
                max_new = None
            # Trace-mode answers end with attached punctuation ("g3."), so the "." stop token
            # never fires; stop at the <eos> the model emits after its answer instead (the
            # scorer cuts at <eos> anyway — this just stops burning budget past it).
            res = evaluate_task(backend, spec, split="test", n=eval_n, length=L,
                                max_new_tokens=max_new, stop_at="<eos>" if emits_trace else ".")
            # In trace mode the answer is the LAST len(gold) tokens of the generated
            # scratchpad, so last_n (not the canonical prefix match) is the fair overall score.
            overall = res["metrics"]["last_n"]["overall"] if emits_trace else res["overall"]
            cell.update(prefix_decomp(res["examples"], trace_mode=emits_trace))
            records = [(i, g, p, ok, None, None)
                       for i, (_pr, g, p, ok) in enumerate(res["examples"])]
        cell["overall"] = overall
        cell["floors"] = cell_floors(spec, L, eval_n)
        # Held-out loss on the SAME document construction the run trained on.
        heldout = build_docs(TK.generate(spec, "test", n=min(eval_n, 200), length=L),
                             use_trace, interleaved)
        cell["heldout_loss"] = held_out_loss(model, tok, heldout, device)

        cell["preds_sample"] = [_rec(*rec) for rec in records[:keep_preds]]
        out[str(L)] = cell
        preds += [_rec(*rec, length=L) for rec in records]
    del run["model"], model, backend
    torch.cuda.empty_cache()
    return {"lengths": out, "final_loss": run["final_loss"],
            "loss_curve": run.get("loss_curve", []), "vocab_size": tok.vocab_size,
            "n_train_docs": len(docs), "epochs": round(steps * batch / max(1, len(docs)), 2),
            "_preds": preds}


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return statistics.mean(xs) if xs else None


def aggregate(runs):
    """Per (task, arch, length): the per-seed values plus their summary statistics.

    The per-seed list is the primary record. This task family is bimodal at the emergence
    threshold (see tasks.CANONICAL["composite_copy_v2"]), so a cell where one seed converged
    and the rest floored has to be readable as such: 0.155 / 0.815 / 0.170 renders as
    "0.38±0.31 (0%)" once it is collapsed to a mean and a >=0.9 convergence test, which is
    indistinguishable from three mediocre seeds and reads as a floor.
    """
    from collections import defaultdict
    by = defaultdict(lambda: defaultdict(dict))     # (task,arch)[length][seed] -> cell
    for r in runs:
        key = (r["task"], r["arch"])
        for L, v in r["lengths"].items():
            by[key][L][r["seed"]] = v
    summary = {}
    for (task, arch), lens in by.items():
        summary.setdefault(task, {})[arch] = {}
        for L, per_seed in lens.items():
            seeds = sorted(per_seed)
            ov = [per_seed[s]["overall"] for s in seeds]
            summary[task][arch][L] = {
                "seeds": seeds,
                "per_seed": ov,
                "mean": statistics.mean(ov),
                "std": statistics.pstdev(ov) if len(ov) > 1 else 0.0,
                "min": min(ov), "median": statistics.median(ov), "max": max(ov),
                "n": len(ov),
                "p_converge": sum(1 for x in ov if x >= 0.9) / len(ov),
                "holder_acc": _mean([per_seed[s].get("holder_acc") for s in seeds]),
                "value_acc": _mean([per_seed[s].get("value_acc") for s in seeds]),
                "checkpoint_acc": _mean([per_seed[s].get("checkpoint_acc") for s in seeds]),
                "heldout_loss": _mean([per_seed[s].get("heldout_loss") for s in seeds]),
                "answer_tokens": max((per_seed[s].get("answer_tokens") or 0) for s in seeds),
                # floors are a property of the items, identical across seeds
                "floors": per_seed[seeds[0]].get("floors") or {},
            }
    return summary


def write_markdown(summary, cfg, path):
    lines = [f"# Local sweep — {cfg['tasks']} ", ""]
    lines.append(f"d_model={cfg['d_model']} n_layers={cfg['n_layers']} steps={cfg['steps']} "
                 f"seeds={cfg['seeds']} train_n={cfg['train_n']} eval_n={cfg['eval_n']}")
    lines.append("")
    for task, archs in summary.items():
        spec = TK.spec_for(task)
        lens = [str(L) for L in spec.eval_lengths]
        cells_seen = [ld[L] for ld in archs.values() for L in lens if L in ld]
        multi_token = any(c["answer_tokens"] >= 2 for c in cells_seen)
        guided = any(c["checkpoint_acc"] is not None for c in cells_seen)
        lines.append(f"## {task}  (eval lengths {', '.join(lens)})")
        cols = [f"L{L} per-seed" for L in lens]
        if multi_token:
            cols.append(f"holder/value @L{lens[0]}")
        if guided:
            cols.append(f"checkpoint acc @L{lens[0]}")
        cols.append(f"held-out loss @L{lens[0]}")
        lines.append("| arch | " + " | ".join(cols) + " |")
        lines.append("|" + "---|" * (len(cols) + 1))
        for arch, ld in sorted(archs.items()):
            cells = []
            for L in lens:
                if L not in ld:                 # incremental summary: this cell has not run yet
                    cells.append("—")
                    continue
                d = ld[L]
                per_seed = " ".join(f"{x:.2f}" for x in d["per_seed"])
                cells.append(f"{per_seed}<br>_min {d['min']:.2f} / med {d['median']:.2f} "
                             f"/ max {d['max']:.2f}_")
            d0 = ld.get(lens[0]) or next(iter(ld.values()))
            if multi_token:
                va = "—" if d0["value_acc"] is None else f"{d0['value_acc']:.2f}"
                cells.append(f"{d0['holder_acc']:.2f} / {va}")
            if guided:
                ck = d0["checkpoint_acc"]
                cells.append("—" if ck is None else f"{ck:.2f}")
            hl = d0["heldout_loss"]
            cells.append("—" if hl is None else f"{hl:.3f}")
            lines.append(f"| {arch} | " + " | ".join(cells) + " |")

        # Floor rows: first-class table rows, recomputed from the exact scored items. Which
        # adversary is largest varies by cell, so the rows are ordered by value (largest
        # first) and the operative floor — the max in a column — is the bolded one.
        floor_at = {}                       # name -> {length: value}
        for name in S5_CHAIN_ADVERSARIES:
            vals = {}
            for L in lens:
                v = [ld[L]["floors"].get(name) for ld in archs.values() if L in ld]
                v = [x for x in v if x is not None]
                if v:
                    vals[L] = v[0]
            if vals:
                floor_at[name] = vals
        top = {L: max((v[L] for v in floor_at.values() if L in v), default=None) for L in lens}
        for name in sorted(floor_at, key=lambda n: -max(floor_at[n].values())):
            cells = []
            for L in lens:
                v = floor_at[name].get(L)
                if v is None:
                    cells.append("—")
                else:
                    cells.append(f"**{v:.3f}**" if v == top[L] else f"{v:.3f}")
            cells += ["—"] * (len(cols) - len(lens))
            lines.append(f"| _floor: {name}_ | " + " | ".join(cells) + " |")

        lines.append("")
        lines.append("_Per-seed values, one per seed in seed order: this family is bimodal at "
                     "the emergence threshold, so a mean hides a converged seed. Floor rows are "
                     "shallow-adversary accuracies recomputed from the exact scored items "
                     "(factworld.validity): `initial_map_chase` ignores every event and "
                     "dereferences the stated initial map; `uniform_non_start` is chance given "
                     "that the gated stream never answers the queried agent. A cell's operative "
                     "floor is the LARGEST value in its column (bold) — which adversary that is "
                     "changes with k, depth and length. Held-out loss is nats/token on the same "
                     "document construction the run trained on._")
        lines.append("")
    path.write_text("\n".join(lines))


def main():
    ap = argparse.ArgumentParser(description="Multi-seed local architecture sweep.")
    ap.add_argument("--tasks", default="binding_v2,composite_copy_v2",
                    help="Comma-separated canonical task names (RETIRED names accepted "
                         "for historical reproduction only; see tasks.RETIRED).")
    ap.add_argument("--archs", default="gdp_hybrid,fprm,transformer",
                    help="Comma-separated architectures (gdp_hybrid_shortconv => gdp_hybrid+shortconv).")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--d_model", type=int, default=256)
    ap.add_argument("--n_layers", type=int, default=4)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--train_n", type=int, default=8000)
    ap.add_argument("--eval_n", type=int, default=100, help="Test examples per length.")
    ap.add_argument("--use_trace", action="store_true", help="Append oracle worked-trace (s5/composite).")
    ap.add_argument("--worked_trace", action="store_true",
                    help="Force worked_trace=True on the spec (needed for the composite "
                         "tasks, whose default is False). Implies --use_trace.")
    ap.add_argument("--chain_depth", type=int, default=None,
                    help="Override spec.chain_depth (s5_chain decomposition probes).")
    ap.add_argument("--k", type=int, default=None,
                    help="Override spec.k (calibration sweeps toward the learnable edge).")
    ap.add_argument("--start_trace", action="store_true",
                    help="s5-shaped single-slot checkpoints (spec.start_trace=True; takes "
                         "precedence over event_trace in the trace builder).")
    ap.add_argument("--interleaved", action="store_true",
                    help="Train on interleaved checkpoints (inside the event stream, the s5 "
                         "dense protocol) instead of an appended trace. Use with --start_trace "
                         "and --guided_eval.")
    ap.add_argument("--guided_eval", action="store_true",
                    help="Guided free-run eval (the s5 formation protocol): events are "
                         "teacher-forced, checkpoints and the answer are generated. Required "
                         "for --interleaved arms at chain_depth 1.")
    ap.add_argument("--compact_events", action="store_true",
                    help="s5-style compact event grammar (spec.compact_events=True; local-only "
                         "rendering ablation, issue #31).")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out_prefix", default=None,
                    help="Output prefix (default: results/sweep_<task0>_<timestamp>).")
    a = ap.parse_args()

    tasks = [t.strip() for t in a.tasks.split(",")]
    archs = [x.strip() for x in a.archs.split(",")]

    # Interleaved single-slot supervision at depth 1 is a copy task, not a tracking task: the
    # gold answer IS the last checkpoint, which sits immediately before "what" in every training
    # document (P=1.000), and the free-running eval prompt deletes exactly that token. Trained
    # under the copy rule, scored with the source removed — the cell has no reading. Guided
    # free-run eval (events forced, checkpoints generated) is the protocol that makes the arm
    # measure state tracking, so require it rather than silently producing an uninterpretable
    # number. See guided_free_run_eval.
    depth = a.chain_depth if a.chain_depth is not None else None
    if a.interleaved and a.start_trace and not a.guided_eval:
        depths = {depth if depth is not None else TK.spec_for(t).chain_depth for t in tasks}
        if 1 in depths:
            ap.error("interleaved + start_trace at chain_depth 1 needs --guided_eval: the "
                     "answer is a verbatim copy of the checkpoint preceding 'what', and the "
                     "free-running eval prompt deletes it, so the cell measures nothing.")
    if a.guided_eval and not (a.interleaved and a.start_trace):
        ap.error("--guided_eval needs --interleaved --start_trace "
                 "(it generates the interleaved checkpoint slots).")

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    prefix = a.out_prefix or f"results/sweep_{tasks[0]}_{ts}"
    from pathlib import Path
    log_path = Path(f"{prefix}.jsonl")
    preds_path = Path(f"{prefix}.preds.jsonl")
    md_path = Path(f"{prefix}.md")
    json_path = Path(f"{prefix}.json")
    log_path.parent.mkdir(parents=True, exist_ok=True)

    cfg = {"tasks": tasks, "archs": archs, "seeds": a.seeds, "steps": a.steps,
           "d_model": a.d_model, "n_layers": a.n_layers, "train_n": a.train_n, "eval_n": a.eval_n,
           "batch": a.batch}
    if a.chain_depth is not None:
        cfg["chain_depth"] = a.chain_depth
    if a.k is not None:
        cfg["k"] = a.k
    if a.interleaved:
        cfg["interleaved"] = True
    if a.guided_eval:
        cfg["guided_eval"] = True
    if a.compact_events:
        cfg["compact_events"] = True

    runs = []
    total = len(tasks) * len(archs) * len(a.seeds)
    done = 0
    print(f"=== sweep: {total} runs -> {log_path} ===", flush=True)
    for task in tasks:
        spec = TK.spec_for(task)
        if a.worked_trace:
            spec = spec.scaled(worked_trace=True)
        if a.chain_depth is not None:
            spec = spec.scaled(chain_depth=a.chain_depth)
        if a.k is not None:
            spec = spec.scaled(k=a.k)
        if a.start_trace:
            spec = spec.scaled(start_trace=True)
        if a.compact_events:
            spec = spec.scaled(compact_events=True)
        for arch in archs:
            use_short, resolved = False, arch
            if arch == "gdp_hybrid_shortconv":
                resolved, use_short = "gdp_hybrid", True
            for seed in a.seeds:
                tag = f"{task} | {arch} | seed {seed}"
                print(f"\n--- [{done+1}/{total}] {tag} ---", flush=True)
                try:
                    r = run_one(spec, resolved, seed, d_model=a.d_model, n_layers=a.n_layers,
                                steps=a.steps, batch=a.batch, train_n=a.train_n, eval_n=a.eval_n,
                                use_short_conv=use_short,
                                use_trace=(a.use_trace or a.worked_trace), device=a.device,
                                interleaved=a.interleaved, guided_eval=a.guided_eval)
                except Exception as e:  # noqa: BLE001
                    import traceback; traceback.print_exc()
                    r = {"error": str(e)}
                run_preds = r.pop("_preds", [])
                rec = {"task": task, "arch": arch, "seed": seed, **cfg, **r}
                with log_path.open("a") as f:
                    f.write(json.dumps(rec) + "\n")
                if run_preds:
                    with preds_path.open("a") as f:
                        for p in run_preds:
                            f.write(json.dumps({"task": task, "arch": arch, "seed": seed, **p}) + "\n")
                if "lengths" in r:
                    runs.append(rec)
                    ov = {L: v["overall"] for L, v in r["lengths"].items()}
                    fl = {L: round(v["heldout_loss"], 3) for L, v in r["lengths"].items()}
                    # the operative floor (max over adversaries), so the console line is
                    # readable without looking the cell up in the markdown
                    fo = {L: operative_floor(v["floors"]) for L, v in r["lengths"].items()}
                    fo = {L: round(x, 3) for L, x in fo.items() if x is not None}
                    print(f"    -> {ov} floor={fo} "
                          f"train_loss={r.get('final_loss'):.3f} heldout={fl}", flush=True)
                done += 1
                # incremental summary
                summary = aggregate(runs)
                write_markdown(summary, cfg, md_path)
                json_path.write_text(json.dumps({"cfg": cfg, "summary": summary, "runs": runs}, indent=2))
    print(f"\n=== done: {md_path} (predictions: {preds_path}) ===", flush=True)


if __name__ == "__main__":
    main()

"""WHY THE COMPOSED PAD IS CAPPED — the four levers, measured on the pad itself.

The bounded-pad grid leaves ONE number blocking a composition claim: the composed cell's
``slot_acc`` never exceeds 0.85 and DEGRADES during the readout stage on every seed, while both
components sit at 0.99-1.00. Until it reaches component levels with the answer still at floor,
"composition fails" and "the model cannot write the scratchpad the protocol requires" are the same
measurement, and only the second is supported. This module separates the candidate explanations,
each as a number rather than an argument.

    ``--counts``      IS THE COMPOSED PAD SIMPLY LONGER? Pure tokenizer arithmetic, no model: pad
                      tokens, prompt tokens and document tokens per cell at every registered
                      length. The components' TOKEN-MATCHED lengths are in the same table, and they
                      are the cells that decide it: state@80 and bind@132 carry LONGER pads than
                      composed@48 and read 1.000.

    ``--forced``      IS THE PER-EVENT UPDATE MISSING, OR ONLY THE CLOSED LOOP? The guided read
                      free-runs — every pad token the model writes is fed back — so one wrong
                      slot corrupts every later block, and a capped free-run accuracy is equally
                      consistent with "never learned the update" and "learned it, cannot survive
                      its own errors". This forces the GOLD pad and reads the same slots in one
                      pass. It is a DIAGNOSTIC: teacher-forced accuracy is not a score on the task,
                      because the true history is exactly what the task withholds.

    ``--decompose``   WHERE in the pad the errors are. The same free-running read the grid scores,
                      instrumented per slot rather than pooled:
                        by ORDINAL   accuracy against event index, as a fraction of the stream. A
                                     flat profile is a systematic gap; a decaying one is drift.
                        by POSITION  block token 0 (the moved/resolved value) against token 1 (the
                                     displaced value).
                        by OPERAND   events whose operand is NAMED against events whose operand is
                                     RESOLVED through the other structure. Only the composed cell
                                     has the second kind, and it is the cell's defining work, so
                                     this is the one split that can localise the gap to the
                                     composition rather than to the format.
                        FIRST ERROR  the ordinal of each item's first wrong slot, which says
                                     whether items fail from the start or survive a while.

    ``--overwrite``   IS THE PAD BEING OVERWRITTEN BY ANSWER SUPERVISION? Reads a checkpoint under
                      three mixes for a short, equal number of steps — the registered mix, that
                      mix with the answer-masked copy dropped, and the composed cell alone — and
                      reports composed ``slot_acc`` before and after each. If the pad degrades only
                      where answer documents are present, the degradation is the answer stage
                      trading the scratchpad away; if it degrades under all three, it is not.

Usage:
    .venv-train/bin/python scripts/probe_s5bind_v3_composed_pad_20260802.py --counts \\
        --out results/20260802_composed_pad_counts.json
    ... --forced --ckpt_dir results/<run>_ckpt --seeds 0 1 2 --n 512
    ... --decompose --ckpt_dir results/<run>_ckpt --seeds 0 1 2 --n 512
    ... --overwrite --ckpt_dir results/<run>_ckpt --seeds 0 1 2 --steps 600
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts"))
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from factworld import tasks as TK                                          # noqa: E402
from factworld import validity as V                                        # noqa: E402
from factworld.composition import SWAP, read as _read                      # noqa: E402
from factworld.tokenizer import Tokenizer                                  # noqa: E402
import protocol_s5bind_v3_three_cell_20260731 as P                         # noqa: E402
import experiment_s5bind_v3_three_cell_local_20260731 as E                 # noqa: E402
import experiment_s5bind_v3_bounded_pad_20260802 as B                      # noqa: E402

# Every cell at its registered lengths, plus each component's TOKEN-MATCHED control — the cells
# that decide the length lever, because they are longer than the composed cell they control.
CELLS = (("state", 17), ("state", 23), ("state", 34), ("state", 80), ("state", 108),
         ("bind", 31), ("bind", 41), ("bind", 62), ("bind", 132),
         ("composed", 48), ("composed", 64), ("composed", 96))


def _tok_and_specs():
    specs = E.three_cell_specs(P.TRAIN_LENGTHS)
    world, r = TK.build_world(specs["composed"])
    return Tokenizer.build([world], r), specs


def operand_kinds(ex):
    """``([resolved], [is_swap], [source])`` per event.

    ``resolved`` is True where the operand has to be looked up in a map rather than read off the
    event line. It is 1.000 on the composed cell and 0.000 on both components (``--counts``), so it
    separates the cells and not the events inside one.

    ``is_swap`` is the split that does separate events inside the composed stream: a swap writes
    two POINTER cells and a give writes a HOLDER cell and the value it displaced, so the two halves
    of the composed pad are the two components' own work and can be scored apart.

    ``source`` is the split that separates the composed cell's TWO-HOP TOKEN from the thing it was
    pooled with (``validity.s5_bind_v3_pad_event_source``). A swap's ``swap_p0`` is two dependent
    reads either way, but on a SAME-source swap both are reads of P — which a policy holding P
    alone performs exactly, and which is the state component's own carrier depth — while on a CROSS
    swap the first read is of the holder map and the second of the pointer map. Only the second is
    the composition, and a pooled number cannot tell a model that does one from a model that does
    both.
    """
    rec = _read(ex.prompt)
    if rec is None:
        return None, None, None
    return ([src != "N" for _k, _t, _r, src in rec["events"]],
            [kind == SWAP for kind, _t, _r, _s in rec["events"]],
            [V.s5_bind_v3_pad_event_source(kind, src)
             for kind, _t, _r, src in rec["events"]])


# ---- (1) the length lever ----------------------------------------------------------------------
def counts(a):
    """Pad, prompt and document token counts per cell at every registered length. No model."""
    tok, specs = _tok_and_specs()
    rows = []
    for cell, L in CELLS:
        spec = specs[cell]
        ags, obs = B.slot_order(spec)
        ex = TK.generate(spec, "test", n=a.n_count, length=L)
        pad_slots, doc_toks, prompt_toks, resolved, n_ev = [], [], [], [], []
        for e in ex:
            got = B.narrow_interleaved(e, a.format, ags, obs)
            if got is None:
                continue
            toks, slots, _gold = got
            pad_slots.append(len(slots))
            doc_toks.append(len(tok.encode(" ".join(toks) + " " + e.answer, add_eos=True)))
            prompt_toks.append(len(tok.encode(" ".join(toks))))
            ks, _sw, _src = operand_kinds(e)
            if ks is not None:
                resolved.append(sum(ks) / max(1, len(ks)))
                n_ev.append(len(ks))
        mean = lambda v: (sum(v) / len(v) if v else None)          # noqa: E731
        rows.append({"cell": cell, "L": L, "n": len(pad_slots),
                     "events": mean(n_ev), "pad_tokens": mean(pad_slots),
                     "prompt_tokens": mean(prompt_toks), "doc_tokens": mean(doc_toks),
                     "pad_share": (mean(pad_slots) / mean(prompt_toks)
                                   if pad_slots and mean(prompt_toks) else None),
                     "resolved_operand_frac": mean(resolved)})
        r = rows[-1]
        print(f"  {cell:9s}@{L:<4d} events={r['events']:6.1f}  pad={r['pad_tokens']:7.1f}  "
              f"prompt={r['prompt_tokens']:7.1f}  doc={r['doc_tokens']:7.1f}  "
              f"pad/prompt={r['pad_share']:.3f}  resolved_operand="
              f"{r['resolved_operand_frac']:.3f}", flush=True)
    return {"cells": rows, "format": a.format, "pad_width": B.PAD_WIDTH[a.format]}


# ---- (2) the closed loop -----------------------------------------------------------------------
def teacher_forced(model, tok, spec, L, n, device, fmt, batch=16):
    """Slot accuracy with the GOLD pad in context: one forward per item, argmax at each slot.

    Directly comparable with the free-running ``slot_acc`` the grid scores, and the gap between
    them is the cost of feeding the model its own writes.
    """
    import torch

    ags, obs = B.slot_order(spec)
    ex = TK.generate(spec, "test", n=n, length=L)
    prepped = []
    for e in ex:
        got = B.narrow_interleaved(e, fmt, ags, obs)
        if got is None:
            return None
        toks, slots, gold = got
        ids, at = [], []
        for j, t in enumerate(toks):
            if j in set(slots):
                at.append(len(ids))
            ids += tok.encode(t)
        prepped.append((ids, at, gold, slot_layout(e, fmt), *operand_kinds(e)))
    hit = tot = 0
    by_pos = [[0, 0], [0, 0]]
    by_named = [[0, 0], [0, 0]]                      # [named, resolved] x [hit, tot]
    by_kind = [[0, 0], [0, 0]]                       # [give, swap] x [hit, tot]
    by_kp = [[0, 0] for _ in KIND_POS]               # give_p0, give_p1, swap_p0, swap_p1
    by_kps = {p: [0, 0] for p in KIND_POS_SRC}       # ... each split by the event's source class
    model.eval()
    with torch.no_grad():
        for b0 in range(0, len(prepped), batch):
            chunk = prepped[b0:b0 + batch]
            ml = max(len(c[0]) for c in chunk)
            inp = torch.full((len(chunk), ml), tok.pad_id, dtype=torch.long, device=device)
            for ri, (ids, *_x) in enumerate(chunk):
                inp[ri, :len(ids)] = torch.tensor(ids, device=device)
            with torch.autocast(device, dtype=torch.bfloat16):
                pred = model(inp).argmax(-1)
            for ri, (ids, at, gold, layout, kinds, swaps, srcs) in enumerate(chunk):
                for si, (pos, g) in enumerate(zip(at, gold)):
                    ev, cell = layout[si]
                    q = 0 if cell.endswith("p0") else 1
                    ok = int(tok.id_to_token.get(int(pred[ri, pos - 1]), "") == g)
                    hit += ok
                    tot += 1
                    by_pos[q][0] += ok
                    by_pos[q][1] += 1
                    if kinds is not None:
                        k = 1 if kinds[ev] else 0
                        by_named[k][0] += ok
                        by_named[k][1] += 1
                        s = 1 if swaps[ev] else 0
                        by_kind[s][0] += ok
                        by_kind[s][1] += 1
                        by_kp[KIND_POS.index(cell)][0] += ok
                        by_kp[KIND_POS.index(cell)][1] += 1
                        key = f"{cell}|{srcs[ev]}"
                        by_kps[key][0] += ok
                        by_kps[key][1] += 1
    return {"per_slot": hit / max(1, tot), "n_slots": tot, "n": len(prepped),
            "by_position": [c[0] / c[1] if c[1] else None for c in by_pos],
            "by_operand": _split(by_named, ("named", "resolved")),
            "by_event": _split(by_kind, ("give", "swap")),
            "by_kind_position": _split(by_kp, KIND_POS),
            "by_kind_position_source": _split([by_kps[p] for p in KIND_POS_SRC], KIND_POS_SRC),
            "hops": HOPS}


def _split(counts_, names):
    """``{name: accuracy, n_name: count}`` for a partition of the scored slots."""
    out = {}
    for nm, c in zip(names, counts_):
        out[nm] = (c[0] / c[1]) if c[1] else None
        out[f"n_{nm}"] = c[1]
    return out


# THE HOP COUNT OF EACH PAD TOKEN ON THE COMPOSED CELL, which is what the (kind, position) cell
# names THERE and only there. ``pad_values`` writes, for a SWAP of (tgt, x) with x RESOLVED through
# the holder map, ``[P[x], P[tgt]]`` post-swap: token 0 needs the operand resolved AND then read
# through the pointer map (TWO hops) while token 1 is the displaced pointer cell (one). A GIVE
# writes ``[x, old B[tgt]]``: the resolved operand and the displaced holder, one hop each. So
# exactly one of the four cells is the two-hop update, and it is the composed cell's defining work.
#
# ON A COMPONENT CELL EVERY OPERAND IS NAMED (``--counts``: resolved fraction 0.000 against the
# composed cell's 1.000), so the state component's ``swap_p0`` is a ONE-hop read of P and carries
# no second hop to fail at. The hop labels below therefore apply to the composed rows; a component
# row's ``swap_p0`` at 1.000 is not the same quantity and must not be read as one.
def slot_layout(ex, fmt):
    """``(event index, pad cell)`` per emitted slot.

    A format whose block width depends on the EVENT KIND (``before2``) has no constant stride from
    slot ordinal to event, so the mapping is read off the format rather than divided out of the
    ordinal. Under a constant-width format this is exactly ``(si // w, KIND_POS[2 * swap + si % w])``.
    """
    rec = _read(ex.prompt)
    if rec is None:
        return None
    return [(i, cell) for i, (kind, *_r) in enumerate(rec["events"])
            for cell in V.s5_bind_v3_pad_cells(kind, fmt)]


KIND_POS = ("give_p0", "give_p1", "swap_p0", "swap_p1")
HOPS = {"give_p0": 1, "give_p1": 1, "swap_p0": 2, "swap_p1": 1}
HOPS_NAMED = {"give_p0": 1, "give_p1": 1, "swap_p0": 1, "swap_p1": 1}
# ... and each of them split by the event's source class, because ``swap_p0|cross`` is the only
# one of the twelve whose write needs BOTH structures. ``swap_p0|same`` is two reads of P, which
# is the state component's own work at depth 2, and pooling the two makes a model that does the
# same-source write and floors on the cross one read like a model that composes.
KIND_POS_SRC = tuple(f"{p}|{s}" for p in KIND_POS for s in V.S5_BIND_V3_PAD_SOURCES)


# ---- (3) where the free-running pad breaks -----------------------------------------------------
def decompose_free_run(model, tok, spec, L, n, device, fmt, batch=128):
    """The grid's own free-running read, instrumented per slot instead of pooled.

    Same decode as ``B.bounded_free_run_batched`` — events teacher-forced, every pad token
    generated and fed back — so the pooled ``per_slot`` it returns reproduces the grid's
    ``slot_acc`` exactly, and every breakdown is a partition of that number.
    """
    import torch

    ex = TK.generate(spec, "test", n=n, length=L)
    ags, obs = B.slot_order(spec)
    prepped = []
    for e in ex:
        got = B.narrow_interleaved(e, fmt, ags, obs)
        if got is None:
            return None
        toks, slots, gold = got
        prepped.append((toks, slots, set(slots), gold, slot_layout(e, fmt), *operand_kinds(e)))
    n_slots = len(prepped[0][1])
    n_ev = 1 + max(ev for _t, _s, _ss, _g, lay, *_r in prepped for ev, _c in lay)
    hit = tot = 0
    by_ord = [[0, 0] for _ in range(n_ev)]
    by_pos = [[0, 0], [0, 0]]
    by_named = [[0, 0], [0, 0]]
    by_kind = [[0, 0], [0, 0]]
    by_kp = [[0, 0] for _ in KIND_POS]
    by_kps = {p: [0, 0] for p in KIND_POS_SRC}
    first_err, perfect = [], 0
    model.eval()
    with torch.no_grad():
        for b0 in range(0, len(prepped), batch):
            chunk = prepped[b0:b0 + batch]
            ids = [[] for _ in chunk]
            cursor = [0] * len(chunk)
            gen = [[] for _ in chunk]
            for ordinal in range(n_slots):
                for i, (toks, slots, slotset, _g, _lay, _k, _s, _src) in enumerate(chunk):
                    while cursor[i] < slots[ordinal]:
                        if cursor[i] not in slotset:
                            ids[i] += tok.encode(toks[cursor[i]])
                        cursor[i] += 1
                for i, tid in enumerate(E._batched_argmax(model, ids, tok, device)):
                    ids[i].append(tid)
                    gen[i].append(tok.id_to_token.get(tid, "<unk>"))
                    cursor[i] += 1
            for i, (_t, _s, _ss, gold, layout, kinds, swaps, srcs) in enumerate(chunk):
                fe = None
                for si, (g, p) in enumerate(zip(gold, gen[i])):
                    ev, cell = layout[si]
                    q = 0 if cell.endswith("p0") else 1
                    ok = int(g == p)
                    hit += ok
                    tot += 1
                    by_ord[ev][0] += ok
                    by_ord[ev][1] += 1
                    by_pos[q][0] += ok
                    by_pos[q][1] += 1
                    if kinds is not None:
                        kk = 1 if kinds[ev] else 0
                        by_named[kk][0] += ok
                        by_named[kk][1] += 1
                        sw = 1 if swaps[ev] else 0
                        by_kind[sw][0] += ok
                        by_kind[sw][1] += 1
                        by_kp[KIND_POS.index(cell)][0] += ok
                        by_kp[KIND_POS.index(cell)][1] += 1
                        key = f"{cell}|{srcs[ev]}"
                        by_kps[key][0] += ok
                        by_kps[key][1] += 1
                    if not ok and fe is None:
                        fe = ev
                perfect += int(fe is None)
                first_err.append(-1 if fe is None else fe)
    model.train()
    q = sorted(x for x in first_err if x >= 0)
    return {"per_slot": hit / max(1, tot), "n": len(prepped), "n_events": n_ev,
            "by_ordinal": [c[0] / c[1] if c[1] else None for c in by_ord],
            "by_position": [c[0] / c[1] if c[1] else None for c in by_pos],
            "by_operand": _split(by_named, ("named", "resolved")),
            "by_event": _split(by_kind, ("give", "swap")),
            "by_kind_position": _split(by_kp, KIND_POS),
            "by_kind_position_source": _split([by_kps[p] for p in KIND_POS_SRC], KIND_POS_SRC),
            "hops": HOPS,
            "items_perfect": perfect / max(1, len(prepped)),
            "first_error_median": (q[len(q) // 2] if q else None),
            "first_error_hist": dict(Counter(min(x, n_ev) for x in first_err).most_common(10))}


# ---- (4) is the answer supervision overwriting the pad? ----------------------------------------
MIXES = {
    "registered": dict(answer_docs=True, mix=None),
    "no_answer_docs": dict(answer_docs=False, mix=None),
    "composed_only": dict(answer_docs=True, mix={"composed": 1.0}),
}


def overwrite(a):
    """Short equal-length continuations under three mixes, composed ``slot_acc`` before/after."""
    import torch
    from factworld import train as T

    tok, specs = _tok_and_specs()
    cells = [(c.split("@")[0], int(c.split("@")[1])) for c in a.cells.split(",")]
    out = {"generated": datetime.now(timezone.utc).isoformat(), "cfg": vars(a), "runs": []}
    want = [x for x in (a.mixes or ",".join(MIXES)).split(",") if x]
    for seed in a.seeds:
        pth = E.checkpoint_path(a.ckpt_dir, a.arch, seed)
        if not Path(pth).exists():
            print(f"  -- no checkpoint {pth}; skipped", flush=True)
            continue
        for name, m in ((k, MIXES[k]) for k in want):
            model, blob = E.load_checkpoint(pth, a.device)
            before = {f"{c}@{L}": B.bounded_free_run_batched(
                model, tok, specs[c], L, a.n, a.device, a.format, batch=a.batch_read)[1]
                for c, L in cells}
            docs, plens = B.stage_documents(specs, m["mix"] or dict(E.SCHEDULE[-1][2]),
                                            a.train_n, tok, a.format,
                                            answer_docs=m["answer_docs"], answer_ratio=1)
            t0 = time.time()
            run = T.run(a.arch, tok, docs, [], steps=a.steps, batch=a.batch,
                        d_model=blob["build"]["d_model"], n_layers=blob["build"]["n_layers"],
                        n_heads=blob["build"]["n_heads"], d_ff=blob["build"]["d_ff"],
                        lr=a.lr, seed=seed, return_model=True, device=a.device, model=model,
                        use_short_conv=True, prompt_lens=plens, warmup=a.warmup)
            model = run["model"]
            after = {f"{c}@{L}": B.bounded_free_run_batched(
                model, tok, specs[c], L, a.n, a.device, a.format, batch=a.batch_read)[1]
                for c, L in cells}
            print(f"  s{seed} {name:16s} " + "  ".join(
                f"{k} {before[k]:.3f}->{after[k]:.3f}" for k in before)
                + f"  [{time.time() - t0:.0f}s]", flush=True)
            out["runs"].append({"seed": seed, "mix": name, "n_docs": len(docs),
                                "before": before, "after": after,
                                "final_loss": run["final_loss"]})
            del model
            torch.cuda.empty_cache()
            with open(a.out, "w") as f:
                json.dump(out, f, indent=1, default=float)
    return out


def model_probe(a, fn, label):
    """Run one per-checkpoint probe over the requested seeds and cells."""
    import torch

    tok, specs = _tok_and_specs()
    cells = [(c.split("@")[0], int(c.split("@")[1])) for c in a.cells.split(",")]
    out = {"generated": datetime.now(timezone.utc).isoformat(), "probe": label,
           "cfg": vars(a), "rows": []}
    for seed in a.seeds:
        pth = E.checkpoint_path(a.ckpt_dir, a.arch, seed)
        if not Path(pth).exists():
            print(f"  -- no checkpoint {pth}; skipped", flush=True)
            continue
        model, _b = E.load_checkpoint(pth, a.device)
        for cell, L in cells:
            t0 = time.time()
            got = fn(model, tok, specs[cell], L, a.n, a.device, a.format,
                     batch=(a.batch_read if label == "free_run" else a.batch_forced))
            got.update({"seed": seed, "cell": cell, "L": L})
            out["rows"].append(got)
            kp = got["by_kind_position"]
            fmt3 = lambda v: "—" if v is None else f"{v:.3f}"       # noqa: E731
            print(f"  s{seed} {cell:9s}@{L:<4d} per_slot={got['per_slot']:.3f}  "
                  + "  ".join(f"{k}({HOPS[k]}h)={fmt3(kp[k])}" for k in KIND_POS)
                  + f"  [{time.time() - t0:.0f}s]", flush=True)
            with open(a.out, "w") as f:
                json.dump(out, f, indent=1, default=float)
        del model
        torch.cuda.empty_cache()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--counts", action="store_true")
    ap.add_argument("--forced", action="store_true")
    ap.add_argument("--decompose", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--ckpt_dir", default=None)
    ap.add_argument("--arch", default="gdp_hybrid")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--cells", default="composed@48,composed@64,composed@96")
    ap.add_argument("--format", default="moved2")
    ap.add_argument("--n", type=int, default=512)
    ap.add_argument("--n_count", type=int, default=200)
    ap.add_argument("--batch_read", type=int, default=128)
    ap.add_argument("--batch_forced", type=int, default=16)
    ap.add_argument("--mixes", default=None,
                    help=f"--overwrite arms to run, comma-separated ({', '.join(MIXES)})")
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--train_n", type=int, default=20000)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="results/20260802_composed_pad_probe.json")
    a = ap.parse_args()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    if a.counts:
        got = counts(a)
        with open(a.out, "w") as f:
            json.dump(got, f, indent=1, default=float)
    elif a.forced:
        model_probe(a, teacher_forced, "teacher_forced")
    elif a.decompose:
        model_probe(a, decompose_free_run, "free_run")
    elif a.overwrite:
        overwrite(a)
    else:
        ap.error("one of --counts / --forced / --decompose / --overwrite")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()

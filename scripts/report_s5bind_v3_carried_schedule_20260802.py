"""THE TRANSFORMER NULL, WITH THE SCHEDULE RESTARTED AND WITH IT CARRIED — side by side.

Both arms are the SAME 25,000 steps over the SAME three-stage curriculum on the SAME documents at
the same width, depth, batch, lr and seeds, read on the same grid against the same floors by the
same code. The one difference is that the control runs the curriculum as three ``train.run`` calls
that each build a fresh AdamW and restart warmup+cosine at 1e-3, and the treatment carries one
optimizer and one schedule across the three stages.

WHAT A DIFFERENCE HERE MEANS. If the treatment lifts the transformer off the floor, the published
transformer null is a property of the training recipe and not of the architecture, and every other
arm's numbers were taken under the same restarting recipe — so the comparison that stands is all
three architectures under ONE schedule, which this round has not bought. If the treatment does not
lift it, the null is an architecture result AT THIS (width, depth, budget, recipe) and can be
reported as one.

Per seed, never a mean. Every cell is printed against the floor the registered run read it
against, and the floors are the registered run's own — not recomputed here, so the two arms are
read against identical numbers.

Usage:
    .venv-train/bin/python scripts/report_s5bind_v3_carried_schedule_20260802.py \
        --control results/s5bind_v3_three_cell_depthmatched_20260801.json \
        --carried results/s5bind_v3_carried_schedule_20260802.json \
        --control_probe results/s5bind_v3_teacher_forced_slots_20260801.json \
        --carried_probe results/s5bind_v3_teacher_forced_slots_carried_20260802.json \
        --out results/s5bind_v3_carried_schedule_20260802_comparison.md
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts"))

import protocol_s5bind_v3_three_cell_20260731 as P                         # noqa: E402
from decode_s5bind_v3_trace_ladder_20260801 import fisher_2x2              # noqa: E402

ARCH = "transformer"
GUIDED_CELLS = (("state", 17), ("state", 80), ("bind", 31), ("bind", 132), ("composed", 48))
PLAIN_CELLS = (("state", 16), ("state", 17), ("state", 23), ("state", 34), ("state", 80),
               ("bind", 16), ("bind", 31), ("bind", 41), ("bind", 62), ("bind", 132),
               ("composed", 16), ("composed", 48), ("composed", 64), ("composed", 96))


def schedule_table(steps, lr, warmup=1000):
    """The two learning-rate schedules, per stage, from the code that produces them."""
    import math
    import statistics as st

    import experiment_s5bind_v3_three_cell_local_20260731 as E

    ss = [max(1, int(round(steps * share))) for _n, share, _w in E.SCHEDULE]

    def mult(g, total):
        if g < warmup:
            return (g + 1) / max(1, warmup)
        return 0.5 * (1 + math.cos(math.pi * min(1.0, (g - warmup) / max(1, total - warmup))))

    out = ["| stage | steps | control mean lr | control lr at stage start / end | "
           "carried mean lr | carried lr at stage start / end |",
           "|---|---|---|---|---|---|"]
    off, allc, allk = 0, [], []
    for (name, _share, _w), n in zip(E.SCHEDULE, ss):
        c = [lr * mult(i, n) for i in range(n)]
        k = [lr * mult(off + i, steps) for i in range(n)]
        allc += c
        allk += k
        out.append(f"| {name} | {n} | {st.mean(c):.3e} | {c[0]:.2e} / {c[-1]:.2e} | "
                   f"{st.mean(k):.3e} | {k[0]:.2e} / {k[-1]:.2e} |")
        off += n
    out += [f"| _whole run_ | {sum(ss)} | {st.mean(allc):.4e} | | {st.mean(allk):.4e} | |", "",
            f"The control returns the learning rate to its peak "
            f"{sum(1 for i in range(1, len(allc)) if allc[i] > 0.9 * lr >= allc[i - 1])} times "
            f"and anneals it to ~0 three times; the carried schedule peaks once. The stage that "
            "carries the composition — stage 3, mix 0.70 composed — runs a full 0 -> lr -> 0 "
            f"cycle under the control at mean {st.mean(allc[-ss[-1]:]):.3e} against the decaying "
            f"tail's {st.mean(allk[-ss[-1]:]):.3e} under the carried one. The Adam moments "
            "entering that stage are ZERO under the control — the optimizer is rebuilt — and "
            f"carry {sum(ss[:-1])} steps of history under the carried schedule.", ""]
    return out


def load(path):
    """``(runs by seed, plain floors, guided floors)`` for one arm."""
    d = json.load(open(path))
    runs = {r["seed"]: r for r in d["runs"] if r["arch"] == ARCH}
    return runs, d.get("floors", {}), d.get("guided_floors", {}), d.get("verdicts", {}), d["cfg"]


def plain(run, cell, L):
    v = run["stages"][-1]["eval"].get(cell, {}).get(str(L))
    return None if v is None else float(v)


def load_trace_fallback(path, seeds_runs):
    """The control's TRACE values, keyed ``(seed, cell, L)``, off the ladder decode.

    The control run predates the trace read, so its own record carries ``trace: null`` and the
    trace channel would print as absent for one arm only — which would leave the comparison
    readable on one channel when the standing rule is both. The ladder decoded the SAME
    checkpoints under the SAME guided protocol at the same n, and its answer values are
    bit-identical to the ones in the run's own record (asserted below), so its trace is this
    run's trace and not a substitute measurement. If any answer disagrees the fallback is
    dropped whole rather than used selectively.
    """
    rows = {}
    p = Path(path)
    if not p.exists():
        return {}
    for line in p.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("arch") == ARCH:
            rows[(r["seed"], r["cell"], r["L"])] = r
    for (seed, cell, x), r in rows.items():
        run = seeds_runs.get(seed)
        if run is None:
            continue
        blk = run["stages"][-1]["guided"].get(cell, {}).get(str(x))
        if blk and blk.get("match") is not None and abs(float(blk["match"]) - r["answer"]) > 1e-9:
            return {}
    return rows


def guided(run, cell, L, read, fallback=None, seed=None):
    blk = run["stages"][-1]["guided"].get(cell, {}).get(str(L))
    if not blk:
        return None, 0
    if read == "answer":
        return (None if blk.get("match") is None else float(blk["match"])), blk.get("n", P.N_GUIDED)
    tr = blk.get("trace")
    if not tr and fallback is not None:
        r = fallback.get((seed, cell, L))
        tr = (r or {}).get("trace")
    return (None if not tr else float(tr["match"])), (tr or {}).get("n", 0)


def mark(v, floor, n):
    if v is None:
        return "—"
    if floor is None:
        return f"{v:.3f}†"
    return f"**{v:.3f}**" if P.clears(v, floor, n)[0] else f"{v:.3f}"


def cmp_row(a, b, na, nb):
    """``(delta string, p string)`` for two proportions on independent item sets."""
    if a is None or b is None or not na or not nb:
        return "—", "—"
    ah, bh = round(a * na), round(b * nb)
    return f"{b - a:+.3f}", f"{fisher_2x2(ah, na - ah, bh, nb - bh):.2g}"


def main():
    ap = argparse.ArgumentParser(description="Restarting vs carried schedule, side by side.")
    ap.add_argument("--control", default="results/s5bind_v3_three_cell_depthmatched_20260801.json")
    ap.add_argument("--carried", default="results/s5bind_v3_carried_schedule_20260802.json")
    ap.add_argument("--control_probe",
                    default="results/s5bind_v3_teacher_forced_slots_20260801.json")
    ap.add_argument("--carried_probe",
                    default="results/s5bind_v3_teacher_forced_slots_carried_20260802.json")
    ap.add_argument("--out", default="results/s5bind_v3_carried_schedule_20260802_comparison.md")
    ap.add_argument("--control_trace_jsonl",
                    default="results/s5bind_v3_trace_ladder_20260801.jsonl",
                    help="the ladder decode off the control's OWN checkpoints, which carries the "
                         "trace channel the control run predates. Used only where the control "
                         "record has no trace, and only after every shared answer value is "
                         "verified bit-identical.")
    a = ap.parse_args()

    cr, cfl, cgf, cvd, ccfg = load(a.control)
    tr, _tfl, _tgf, tvd, tcfg = load(a.carried)
    ctrace = load_trace_fallback(a.control_trace_jsonl, cr)
    seeds = sorted(set(cr) & set(tr))
    n_plain, n_guided = ccfg["eval_n"], ccfg["guided_n"]

    L = ["# The transformer null: schedule restarted per stage vs carried across stages", "",
         f"`{ARCH}` · d_model {ccfg['d_model']} · {ccfg['n_layers']} layers · "
         f"{ccfg['steps']} steps · batch {ccfg['batch']} · lr {ccfg['lr']} · "
         f"{ccfg['train_n']} items per stage · seeds {seeds} · k=6 · informed chance 0.200 · "
         "match.", "",
         "**CONTROL** (`" + Path(a.control).name + "`): three `train.run` calls, one per stage. "
         "Each builds a fresh AdamW and restarts warmup+cosine at lr, so the learning rate goes "
         "0 → lr → 0 three times and the Adam moments are discarded twice.", "",
         "**CARRIED** (`" + Path(a.carried).name + "`): one optimizer and one warmup+cosine over "
         f"the global {tcfg['steps']} steps. Same documents in the same order (they are cached and "
         "deterministic), same mixes, same step shares, same seeds, same eval grid, same floors.",
         "",
         "A **bold** cell clears its own floor under the pre-registered rule "
         f"(z > {P.Z_CLEAR}, margin >= {P.MARGIN}); a † marks a cell with no floor on that read. "
         "Floors are the registered run's own, so both arms are read against identical numbers. "
         "Per seed, never a mean — this family is bimodal at the emergence threshold.", "",
         "## What the two schedules actually are", "",
         "The mean learning rate over the whole run is the same to four figures under both, so "
         "the treatment is not more learning rate — it is the shape.", ""]
    L += schedule_table(ccfg["steps"], ccfg["lr"])
    L += ["## Verdict, by the pre-registered rule", "",
         "| read | control | carried |", "|---|---|---|"]
    for read in ("plain", "guided"):
        c = (cvd.get(ARCH) or {}).get(read, {})
        t = (tvd.get(ARCH) or {}).get(read, {})
        L.append(f"| {read} | {c.get('verdict', '—')} | {t.get('verdict', '—')} |")
    L += [""]

    # ---- the guided read, both channels
    for read in ("answer", "trace"):
        L += [f"## The GUIDED read, {read.upper()} channel", ""]
        if read == "trace" and ctrace:
            L += [f"The control's trace column is the ladder decode "
                  f"(`{Path(a.control_trace_jsonl).name}`) off the control's OWN checkpoints: the "
                  "control run predates the trace read and its record carries `trace: null`. It "
                  "is the same protocol at the same n on the same weights, and every answer value "
                  "the two records share is bit-identical, so this is that run's trace and not a "
                  "substitute measurement.", ""]
        L += ["| cell | seed | control | carried | difference | p |", "|---|---|---|---|---|---|"]
        for cell, x in GUIDED_CELLS:
            f = cgf.get(f"{cell}@{x}", {}).get("floor")
            for s in seeds:
                cv, cn = guided(cr[s], cell, x, read, ctrace, s)
                tv, tn = guided(tr[s], cell, x, read)
                d, p = cmp_row(cv, tv, cn or n_guided, tn or n_guided)
                L.append(f"| {cell}@{x} | {s} | {mark(cv, f, n_guided)} | "
                         f"{mark(tv, f, n_guided)} | {d} | {p} |")
            pad = cgf.get(f"{cell}@{x}", {}).get("pad_reach")
            L.append(f"| {cell}@{x} | _floor_ | "
                     + (f"{f:.3f}" if f is not None else
                        "unfloorable" + (f" (pad {pad:.3f})" if pad is not None else ""))
                     + " | | | |")
        L += [""]

    # ---- the plain read
    L += ["## The PLAIN read (answer off the plain prompt, one token, n = %d)" % n_plain, "",
          "| cell | seed | control | carried | difference | p |", "|---|---|---|---|---|---|"]
    for cell, x in PLAIN_CELLS:
        f = cfl.get(f"{cell}@{x}", {}).get("floor")
        for s in seeds:
            cv, tv = plain(cr[s], cell, x), plain(tr[s], cell, x)
            d, p = cmp_row(cv, tv, n_plain, n_plain)
            L.append(f"| {cell}@{x} | {s} | {mark(cv, f, n_plain)} | {mark(tv, f, n_plain)} | "
                     f"{d} | {p} |")
        L.append(f"| {cell}@{x} | _floor_ | " + (f"{f:.3f}" if f is not None else "—") + " | | | |")
    L += [""]

    # ---- the teacher-forced probe
    if Path(a.control_probe).exists() and Path(a.carried_probe).exists():
        cp = json.load(open(a.control_probe))["rows"]
        tp = json.load(open(a.carried_probe))["rows"]
        L += ["## The teacher-forced probe — moving slots, gold history", "",
              "The diagnostic that identified the artifact: argmax at every checkpoint slot with "
              "the TRUE history in front of the model, restricted to the slots whose value "
              "differs from the previous checkpoint's, which is the only part a copier does not "
              "get for free. It is not a score on the task — the true history is exactly what "
              "the task withholds — and no verdict reads it.", "",
              "| cell | seed | control | carried | difference |", "|---|---|---|---|---|"]
        for cell, x in (("state", 17), ("state", 80), ("bind", 31), ("bind", 62),
                        ("composed", 48), ("composed", 96)):
            for s in seeds:
                k = f"{ARCH}|{s}|{cell}|{x}"
                c = (cp.get(k) or {}).get("moving_slots")
                t = (tp.get(k) or {}).get("moving_slots")
                L.append(f"| {cell}@{x} | {s} | "
                         + ("—" if c is None else f"{c:.3f}") + " | "
                         + ("—" if t is None else f"{t:.3f}") + " | "
                         + ("—" if c is None or t is None else f"{t - c:+.3f}") + " |")
        L += [""]

    # ---- training loss, since it is the quantity the schedule acts on
    L += ["## Final training loss per stage", "",
          "The schedule acts on the optimisation, so the loss is the first place a difference "
          "would show. Stage 3's loss is on the composition-weighted mix.", "",
          "| seed | stage | control | carried |", "|---|---|---|---|"]
    for s in seeds:
        for i, st in enumerate(cr[s]["stages"]):
            L.append(f"| {s} | {st['stage']} | {st['final_loss']:.4f} | "
                     f"{tr[s]['stages'][i]['final_loss']:.4f} |")
    L += [""]

    Path(a.out).write_text("\n".join(L))
    Path(str(a.out).replace(".md", ".json")).write_text(json.dumps(
        {"written_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
         "control": a.control, "carried": a.carried, "arch": ARCH, "seeds": seeds,
         "guided": {f"{c}@{x}": {read: {str(s): {"control": guided(cr[s], c, x, read, ctrace, s)[0],
                                                 "carried": guided(tr[s], c, x, read)[0]}
                                        for s in seeds}
                                 for read in ("answer", "trace")}
                    for c, x in GUIDED_CELLS},
         "plain": {f"{c}@{x}": {str(s): {"control": plain(cr[s], c, x),
                                         "carried": plain(tr[s], c, x)} for s in seeds}
                   for c, x in PLAIN_CELLS},
         "guided_floors": {k: v.get("floor") for k, v in cgf.items()},
         "plain_floors": {k: v.get("floor") for k, v in cfl.items()},
         "verdicts": {"control": cvd.get(ARCH), "carried": tvd.get(ARCH)}},
        indent=2, default=float))
    print(f"=== wrote {a.out} ===")
    for read in ("plain", "guided"):
        print(f"  {read}: control {(cvd.get(ARCH) or {}).get(read, {}).get('verdict')} -> "
              f"carried {(tvd.get(ARCH) or {}).get(read, {}).get('verdict')}")


if __name__ == "__main__":
    main()

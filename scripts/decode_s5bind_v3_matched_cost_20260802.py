"""COMPLETE THE MATCHED-COST CONTROL — the two rungs the previous decode did not buy.

WHAT WAS MISSING. The separation's claim is "neither length nor depth": the composed cell sits
below its work-matched state component, and it is not because the composed stream is longer. The
control that carries that clause is the component read at the length whose FORWARD PASS costs what
the composed cell's costs (``protocol.TOKEN_MATCHED``). It was bought at ONE of the three rungs —
composed@48 against state@80 — because the guided decode is O(n L^2). One rung is one point, and a
claim of the form "not length" made at one length is a claim with no length axis in it.

WHAT THIS BUYS. The other two rungs, off the SAME nine saved checkpoints:

    composed@64  (930 prompt tokens)  vs  state@108  (915)
    composed@96  (1348)               vs  state@160  (1331)

Both lengths were already in the PLAIN grid of the training run and were never decoded under the
GUIDED protocol, which is the read the ladder is written on. This is a DECODE: nothing trains, the
weights are the ones the registered run wrote, and the decode loop, the floors and the clears rule
are the 2026-08-01 decode module's own.

WHAT A MATCHED-COST ROW MEANS, AND WHAT IT DOES NOT. state@108 costs what composed@64 costs on the
forward pass and carries 4.7x its state depth; if the component at matched cost is at or above the
composed cell, then "the composed cell is harder" is not "the composed stream is longer". It is
still a WITHIN-RUN comparison: the composed cell has no floor under this protocol (the guided
format writes the whole of P then B at every event, handing out the live slots the one-structure
bound prices), so no row here is ever a cleared floor and none is printed as one.

Usage:
    .venv-train/bin/python scripts/decode_s5bind_v3_matched_cost_20260802.py \
        --ckpt_dir results/s5bind_v3_three_cell_depthmatched_20260801_ckpt \
        --seed_jsonl results/s5bind_v3_trace_ladder_20260801.jsonl \
        --out_prefix results/s5bind_v3_matched_cost_20260802
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts"))
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import protocol_s5bind_v3_three_cell_20260731 as P                         # noqa: E402
import decode_s5bind_v3_trace_ladder_20260801 as D                         # noqa: E402

# The completed control: every composed rung against the component at its TOKEN-MATCHED length.
# ``bind`` has no matched-cost partner at 64 or 96 — its sampler cannot fill the floor's item pool
# past BIND_MATCHED_MAX = 144 — so those rows are registered ABSENT rather than substituted by a
# shorter cell that does not match the cost.
MATCHED_RUNGS = tuple((cl, P.TOKEN_MATCHED[cl]["state"], P.TOKEN_MATCHED[cl]["bind"])
                      for cl in P.LOCAL_LENGTHS)
# Cheapest cell first inside the phase: the decode is O(n L^2) and a run that is cut short must
# leave the cheap rungs measured rather than the expensive ones half-measured.
PHASES = (("ladder", (("state", 17), ("state", 23), ("state", 34),
                      ("bind", 31), ("bind", 41), ("bind", 62),
                      ("composed", 48), ("composed", 64), ("composed", 96))),
          ("matched_cost", (("state", 80), ("state", 108), ("bind", 132), ("state", 160))))
ARCH_ORDER = D.ARCH_ORDER


def _v(done, arch, seed, cell, L, read):
    r = done.get((arch, seed, cell, L))
    if r is None:
        return None
    if read == "answer":
        return r["answer"]
    return None if not r.get("trace") else r["trace"]["match"]


def _n(done, arch, seed, cell, L, read):
    r = done.get((arch, seed, cell, L))
    if r is None:
        return 0
    return r["n"] if read == "answer" else (r.get("trace") or {}).get("n", 0)


def control_rows(done, read):
    """``[(arch, seed, composed_L, state_L, state_v, composed_v, state_n, composed_n)]``."""
    out = []
    for arch in ARCH_ORDER:
        for seed in sorted({k[1] for k in done if k[0] == arch}):
            for cl, sl, _bl in MATCHED_RUNGS:
                if sl is None:
                    continue
                sv, cv = (_v(done, arch, seed, "state", sl, read),
                          _v(done, arch, seed, "composed", cl, read))
                if sv is None or cv is None:
                    continue
                out.append((arch, seed, cl, sl, sv, cv,
                            _n(done, arch, seed, "state", sl, read),
                            _n(done, arch, seed, "composed", cl, read)))
    return out


def interpretable(done, read, lo=0.30, hi=0.95):
    """(arch, seed, rung) cells where BOTH cells are off ceiling and off floor on ``read``.

    A control row where both cells are pinned is consistent with anything; the verdict is stated
    over the rows that could have come out either way, with the pinned rows counted beside them.
    """
    return [(a, s, cl) for a, s, cl, _sl, sv, cv, _sn, _cn in control_rows(done, read)
            if lo <= sv <= hi and lo <= cv <= hi]


@lru_cache(maxsize=1)
def costs():
    """``{cell@L: (charged steps, prompt tokens)}`` for the rows this report prints."""
    import experiment_s5bind_v3_three_cell_local_20260731 as E
    from factworld import tasks as TK
    from factworld.tokenizer import Tokenizer

    specs = E.three_cell_specs(P.TRAIN_LENGTHS)
    world, renderer = TK.build_world(specs["composed"])
    tok = Tokenizer.build([world], renderer)
    want = {("composed", cl) for cl, _s, _b in MATCHED_RUNGS}
    want |= {("state", sl) for _c, sl, _b in MATCHED_RUNGS if sl}
    want |= {("state", P.WORK_MATCHED[cl]["state"]) for cl, _s, _b in MATCHED_RUNGS}
    return {f"{c}@{L}": P.cell_cost(specs[c], L, tok) for c, L in sorted(want)}


def write_report(done, floors, args):
    n = args.guided_n
    ch = 1.0 / 5
    cc = costs()
    L = ["# s5_bind_v3 — the matched-COST control, completed at all three rungs",
         "",
         f"k=6 · informed chance 1/(k-1) = {ch:.3f} · match · n={n} per cell · GUIDED protocol "
         "(events teacher-forced, every per-event checkpoint and the answer generated) · decoded "
         f"from `{args.ckpt_dir}`, nothing trained.",
         "",
         "The control reads each component at the length whose FORWARD PASS costs what the "
         "composed cell's costs, so \"harder because composed\" is separated from \"harder "
         "because longer\". The previous decode bought it at composed@48 only; the two rungs "
         "added here are the ones that give the clause a length axis.",
         "",
         "| rung | composed prompt tokens | matched component | its prompt tokens | its "
         "work-matched partner |", "|---|---|---|---|---|"]
    for cl, sl, _bl in MATCHED_RUNGS:
        wl = P.WORK_MATCHED[cl]["state"]
        L.append(f"| composed@{cl} | {cc[f'composed@{cl}'][1]} | "
                 + (f"state@{sl} | {cc[f'state@{sl}'][1]} | state@{wl} "
                    f"({cc[f'state@{wl}'][1]} tokens) |" if sl else "— | — | — |"))
    L += ["",
          "`bind` has no matched-cost partner at 64 or 96: its sampler cannot fill the floor's "
          f"own item pool past L={P.BIND_MATCHED_MAX}, so those rows are registered ABSENT rather "
          "than filled with a shorter cell that does not match the cost.",
          "",
          "**Per seed, never a mean** — this family is bimodal at the emergence threshold. A "
          "**bold** cell clears its own floor under the pre-registered rule "
          f"(z > {P.Z_CLEAR}, margin >= {P.MARGIN}, at this read's own n); a † marks a cell with "
          "NO FLOOR. The composed cell is unfloorable under this protocol on BOTH channels, so "
          "every composed column below is † and no row here is a cleared floor: the guided format "
          "writes the whole of P then B at every event and hands out the k + m live slots the "
          "one-structure bound prices, to every policy including the task's own algorithm.",
          ""]

    # ---- the control, per rung, both reads
    for read in ("trace", "answer"):
        L += [f"## The control on the {read.upper()} read", ""]
        for cl, sl, _bl in MATCHED_RUNGS:
            if sl is None:
                continue
            L += [f"### composed@{cl} vs state@{sl} "
                  f"({cc[f'composed@{cl}'][1]} against {cc[f'state@{sl}'][1]} prompt tokens)", "",
                  f"| arch | seed | state@{sl} | composed@{cl} | difference | p (this seed) |",
                  "|---|---|---|---|---|---|"]
            for arch in ARCH_ORDER:
                for seed in sorted({k[1] for k in done if k[0] == arch}):
                    sv, cv = (_v(done, arch, seed, "state", sl, read),
                              _v(done, arch, seed, "composed", cl, read))
                    if sv is None or cv is None:
                        L.append(f"| {arch} | {seed} | — | — | — | — |")
                        continue
                    sn = _n(done, arch, seed, "state", sl, read)
                    cn = _n(done, arch, seed, "composed", cl, read)
                    sh, chh = round(sv * sn), round(cv * cn)
                    p = D.fisher_2x2(sh, sn - sh, chh, cn - chh)
                    sf = floors.get(f"state@{sl}", {})
                    cf = floors.get(f"composed@{cl}", {})
                    fk = "answer_floor" if read == "answer" else "trace_floor"
                    L.append(f"| {arch} | {seed} | "
                             f"{D._cell(sv, sf.get(fk), n)} ({sh}/{sn}) | "
                             f"{D._cell(cv, cf.get(fk), n)} ({chh}/{cn}) | "
                             f"{cv - sv:+.3f} | {p:.2g} |")
            sf, cf = floors.get(f"state@{sl}", {}), floors.get(f"composed@{cl}", {})
            fk = "answer_floor" if read == "answer" else "trace_floor"
            pad = cf.get("pad_reach")
            L += ["| _floor_ | | "
                  + (f"{sf[fk]:.3f}" if sf.get(fk) is not None else "—") + " | "
                  + ("unfloorable" + (f" (pad {pad:.3f})" if pad is not None else "")
                     if cf.get(fk) is None else f"{cf[fk]:.3f}")
                  + " | | |", ""]

    # ---- what it settles
    L += ["## What the completed control settles", ""]
    for read in ("trace", "answer"):
        rows = control_rows(done, read)
        held = [r for r in rows if r[4] >= r[5]]
        interp = interpretable(done, read)
        per_rung = []
        for cl, sl, _bl in MATCHED_RUNGS:
            if sl is None:
                continue
            rr = [r for r in rows if r[2] == cl]
            per_rung.append(f"composed@{cl} vs state@{sl}: {sum(1 for r in rr if r[4] >= r[5])}"
                            f"/{len(rr)}")
        L += [f"**{read.upper()} read.** The matched-cost component is at or above the composed "
              f"cell on {len(held)} of the {len(rows)} (architecture, seed, rung) rows "
              f"— {'; '.join(per_rung)}. "
              f"{len(interp)} of those rows have BOTH cells off ceiling and off floor "
              f"({', '.join(f'{a} s{s} @{cl}' for a, s, cl in interp) or 'none'}), and a row with "
              "both cells pinned is consistent with any direction, so it is counted and not "
              "read.",
              ""]
    exc = [(a, s, cl, sl, sv, cv) for a, s, cl, sl, sv, cv, _sn, _cn
           in control_rows(done, "trace") if sv < cv]
    L += ["The exceptions on the trace read are "
          + (", ".join(f"{a} s{s} composed@{cl} {cv:.3f} against state@{sl} {sv:.3f}"
                       for a, s, cl, sl, sv, cv in exc) or "none") + ".",
          "",
          "This is a within-run comparison at every row. The composed cell has no floor under "
          "this protocol, so the control can say the deficit is not explained by prompt length; "
          "it cannot say the composed cell is above any cheap policy.",
          ""]

    # ---- floors
    L += [f"## The floors, recomputed at n = {n} from each cell's own scored items", "",
          "One floor per cell for BOTH channels, because both decode under the same protocol.",
          "",
          "| cell | guided floor (answer & trace) | basis | pad reach | plain-protocol floor | "
          "slot==gold (T1) | copier per-slot | queried slot moves (min/median/max) |",
          "|---|---|---|---|---|---|---|---|"]
    for key in sorted(floors, key=lambda k: (k.split("@")[0], int(k.split("@")[1]))):
        f = floors[key]
        gf = ("**unfloorable**" if f["trace_floor"] is None
              else f"{f['trace_floor']:.3f} ({f['trace_floor'] / ch:.2f}x)")
        mv = f.get("slot_moves", {})
        pr = "—" if f.get("pad_reach") is None else f"{f['pad_reach']:.3f}"
        pf = f.get("answer_floor_plain")
        L.append(f"| {key} | {gf} | {f['trace_basis']} | {pr} | "
                 + ("—" if pf is None else f"{pf:.3f} ({pf / ch:.2f}x)")
                 + f" | {f['slot_is_gold_scored']} | {f.get('copy_per_slot', 0):.3f} | "
                 f"{mv.get('min')}/{mv.get('median')}/{mv.get('max')} |")
    L += ["",
          "Decode batch is recorded per row in the JSON. It is a memory knob and not a scoring "
          "one (right padding, causal models).",
          ""]

    Path(f"{args.out_prefix}.md").write_text("\n".join(L))
    Path(f"{args.out_prefix}.json").write_text(json.dumps(
        {"written_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
         "ckpt_dir": args.ckpt_dir, "n": n, "read": P.TRACE_READ,
         "matched_rungs": [list(r) for r in MATCHED_RUNGS], "cell_costs": cc,
         "floors": floors,
         "rows": [v for _k, v in sorted(done.items(), key=lambda kv: (kv[0][0], kv[0][1],
                                                                     kv[0][2], kv[0][3]))]},
        indent=2, default=float))


def main():
    ap = argparse.ArgumentParser(description="Complete the matched-cost control (decode only).")
    ap.add_argument("--ckpt_dir",
                    default="results/s5bind_v3_three_cell_depthmatched_20260801_ckpt")
    ap.add_argument("--archs", default="gdp_hybrid,fprm,transformer")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--guided_n", type=int, default=P.N_GUIDED)
    ap.add_argument("--guided_batch", type=int, default=32)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out_prefix", default="results/s5bind_v3_matched_cost_20260802")
    ap.add_argument("--seed_jsonl", default="results/s5bind_v3_trace_ladder_20260801.jsonl",
                    help="rows already decoded off the same checkpoints, copied in so they are "
                         "not paid for twice")
    ap.add_argument("--floor_cache", default=None)
    a = ap.parse_args()
    if a.floor_cache is None:
        a.floor_cache = f"{a.out_prefix}_floors.json"

    # Seed the run's own jsonl with the rows the previous decode already bought off these exact
    # weights, so this round pays only for the two missing lengths and the report still holds the
    # whole control.
    jl = Path(f"{a.out_prefix}.jsonl")
    if a.seed_jsonl and Path(a.seed_jsonl).exists() and not jl.exists():
        jl.parent.mkdir(parents=True, exist_ok=True)
        jl.write_text(Path(a.seed_jsonl).read_text())
        print(f"=== seeded {jl} from {a.seed_jsonl} ===", flush=True)
    if a.floor_cache and not Path(a.floor_cache).exists():
        prev = Path("results/s5bind_v3_trace_ladder_floors_20260801.json")
        if prev.exists():
            Path(a.floor_cache).write_text(prev.read_text())

    D.PHASES = PHASES
    D.write_report = write_report
    done, floors = D.decode(a)
    write_report(done, floors, a)
    print(f"\n=== done: {a.out_prefix}.md ===", flush=True)


if __name__ == "__main__":
    main()

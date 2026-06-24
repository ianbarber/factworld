"""Why is post-training state coverage only 1/3 reliable? Base-quality vs post-luck variance decomposition.

post_state.py showed post-training deep-state coverage makes the non-abelian circuit length-general on 1/3 seeds
(s0: L128 0.86; s1/s2 ~floor). To make it RELIABLE we must first know WHERE the fragility lives:
  H1 (base-side): base-circuit QUALITY gates post -> fix = train K bases, SELECT a good one, then post.
  H2 (post-side): post-training optimization LUCK -> fix = best-of-R post restarts + select.

This runs the decomposition: N base seeds, each post-trained R times (different post data-order), all evaluated to
L256. If L128 variance is mostly BETWEEN bases -> H1 (and we check whether a cheap base probe predicts success);
if mostly WITHIN base (same base, different post -> different outcome) -> H2. Also logs a label-free state-norm
growth probe (does post FLATTEN the recurrent-state norm growth past trained depth on successes?).

Reuses: length_mix.train (base), post_state.post_train/make_pool, dense_capstone.e2e_eval/dense_eval,
supervision_sweep.build. Single 3090, ~2.5-3 hr.

  .venv/bin/python followups/non-abelian-state/post_reliability.py
"""
from __future__ import annotations

import copy
import os
import random
import statistics
import sys
from collections import defaultdict
from typing import Any

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from dense_capstone import _world, e2e_eval, dense_eval
from supervision_sweep import build as build_ck             # (w,r,origins,oracle,L,K,rng) -> (words,hidx,vidx)
from post_state import post_train, make_pool                # exact post recipe + short/burnin pools
from length_mix import train                                # base: gdp_hybrid d384x6, lr1e-3, 6000 steps

BASE_SEEDS = [0, 1, 2, 3, 4, 5]
POST_SEEDS = [0, 1, 2]
EVAL_LEN = [16, 24, 32, 64, 128, 256]
SUCCESS_L, SUCCESS_THR = 128, 0.5                            # "length-general" = answer-only L128 > 0.5
N_EVAL = 150
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "post_reliability.md")


def state_norm_ratio(model: Any, tok: Any, w: Any, r: Any, origins: dict) -> dict:
    """Label-free probe: per-(recurrent)-block hidden-norm growth ratio over a length-256 chain
    (mean norm of last 32 positions / first 32). >1 = state norm grows with depth; ~1 = flat/calibrated.

    Args:
        model: the model to probe.
        tok: the atomic tokenizer.
        w: the FactWorld ``World``.
        r: the ``Renderer``.
        origins: the fixed agent -> a0-value map (unused beyond signature parity).

    Returns:
        A dict ``{block_index: norm_growth_ratio}`` (empty if the probe raises).
    """
    try:
        import torch
        caught = {}
        hooks = [blk.register_forward_hook(
            lambda m, i, o, bi=bi: caught.__setitem__(bi, o.detach().float().norm(dim=-1)[0].cpu().tolist()))
            for bi, blk in enumerate(model.blocks)]
        rng = random.Random(123)
        ev = w.sample_hard_chain(256, episode_seed="probe")
        words = ["role", rng.choice(w.roles), "."]
        for h in r.render_history(tuple(ev), with_steps=True):
            words += h.split()
        ids = tok.encode(" ".join(words))
        model.eval()
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            model(torch.tensor([ids], device="cuda"))
        for h in hooks:
            h.remove()
        return {bi: round(statistics.mean(v[-32:]) / max(1e-6, statistics.mean(v[:32])), 2)
                for bi, v in caught.items()}
    except Exception as exc:                                 # probe must never kill the run
        print(f"    norm-probe error: {exc}", flush=True)
        return {}


def main() -> None:
    """Decompose L128 variance into between-base vs within-base across base seeds x post restarts."""
    import torch
    if not torch.cuda.is_available():
        print("no GPU"); return
    from factworld import train as T
    from factworld.oracle import Oracle
    w, r, origins = _world()
    oracle = Oracle(w)
    short_pool = make_pool("short", w, r, origins, oracle, seed=2)
    burn_pool = make_pool("burnin", w, r, origins, oracle, seed=3)
    tok, _, _ = T.prepare(short_pool + burn_pool[:2000], [], [w])
    evs = {L: [build_ck(w, r, origins, oracle, L, 10**9, random.Random(900 + L + j)) for j in range(N_EVAL)]
           for L in EVAL_LEN}
    ckev = [build_ck(w, r, origins, oracle, 16, 1, random.Random(5000 + j)) for j in range(N_EVAL)]  # dense_h@L16
    rows = []
    print("=== POST-RELIABILITY: base-quality vs post-luck variance decomposition ===", flush=True)
    for bs in BASE_SEEDS:
        base = train(tok, short_pool, bs)
        bq = {"L16": e2e_eval(base, tok, w, evs[16])[1],
              "L24": e2e_eval(base, tok, w, evs[24])[1],
              "L32": e2e_eval(base, tok, w, evs[32])[1],
              "dh16": dense_eval(base, tok, w, ckev)[0],
              "norm": state_norm_ratio(base, tok, w, r, origins)}
        print(f"  base s{bs} :: L16={bq['L16']:.2f} L24={bq['L24']:.2f} L32={bq['L32']:.2f} "
              f"dense_h16={bq['dh16']:.2f} norm={bq['norm']}", flush=True)
        for ps in POST_SEEDS:
            m = copy.deepcopy(base)
            post_train(m, tok, burn_pool, 1500, 3e-4, ps)
            acc = {L: e2e_eval(m, tok, w, evs[L])[1] for L in EVAL_LEN}
            rows.append({"base": bs, "post": ps, "bq": bq, "acc": acc,
                         "postnorm": state_norm_ratio(m, tok, w, r, origins)})
            print(f"    base s{bs} post s{ps} :: " + "  ".join(f"L{L}={acc[L]:.2f}" for L in EVAL_LEN)
                  + f"   (success={acc[SUCCESS_L] > SUCCESS_THR})", flush=True)
            del m; torch.cuda.empty_cache()
            write_md(rows)
        del base; torch.cuda.empty_cache()
    write_md(rows)
    print("post_reliability done.", flush=True)


def write_md(rows: list) -> None:
    """Write the base-quality vs post-luck decomposition tables and verdict to ``OUT``."""
    by_base = defaultdict(list)
    for d in rows:
        by_base[d["base"]].append(d)
    lines = [
        "# Post-reliability — base-quality vs post-luck (`post_reliability.py`, 18.5M, 6 bases x 3 posts)\n",
        "Each base seed (short {4,8,16}) post-trained 3x (different post data-order) with deep-state coverage; "
        f"answer-only eval to L256. Success = L{SUCCESS_L} > {SUCCESS_THR}. `dense_h16` = per-step holder acc at "
        "L16 (base-quality probe); `norm` = recurrent hidden-norm growth ratio over L256 (label-free; ~1 = "
        "calibrated). Floor = 0.20.\n",
        "| base | dense_h16 | base_norm(per-block) | "
        + " | ".join(f"post{p}·L{SUCCESS_L}" for p in POST_SEEDS) + " | #succ/3 |",
        "|---|---|---|" + "---|" * (len(POST_SEEDS) + 1),
    ]
    base_means = []
    within_vars = []
    for bs in sorted(by_base):
        ds = sorted(by_base[bs], key=lambda d: d["post"])
        l128 = [d["acc"][SUCCESS_L] for d in ds]
        succ = sum(x > SUCCESS_THR for x in l128)
        bq = ds[0]["bq"]
        nrm = ",".join(str(v) for _, v in sorted(bq["norm"].items())) if bq["norm"] else "—"
        cells = "  ".join(f"{x:.2f}" for x in l128)
        lines.append(f"| s{bs} | {bq['dh16']:.2f} | {nrm} | "
                     + " | ".join(f"{x:.2f}" for x in l128) + f" | {succ} |")
        base_means.append(statistics.mean(l128))
        if len(l128) > 1:
            within_vars.append(statistics.pvariance(l128))
    if base_means:
        between = statistics.pvariance(base_means) if len(base_means) > 1 else 0.0
        within = statistics.mean(within_vars) if within_vars else 0.0
        verdict = ("WITHIN-base dominates -> H2 (post-luck) -> best-of-R post + select"
                   if within > between else
                   "BETWEEN-base dominates -> H1 (base-quality) -> select a good base, then post")
        lines += [
            f"\n**Variance of L{SUCCESS_L}:** between-base = {between:.4f}, within-base = {within:.4f}. "
            f"→ {verdict}.",
            "\n_Full per-(base,post) accuracy:_\n",
            "| base | post | " + " | ".join(f"L{L}" for L in EVAL_LEN) + " | post_norm(per-block) |",
            "|---|---|" + "---|" * (len(EVAL_LEN) + 1),
        ]
        for d in sorted(rows, key=lambda d: (d["base"], d["post"])):
            pn = ",".join(str(v) for _, v in sorted(d["postnorm"].items())) if d["postnorm"] else "—"
            lines.append(f"| s{d['base']} | s{d['post']} | "
                         + " | ".join(f"{d['acc'][L]:.2f}" for L in EVAL_LEN) + f" | {pn} |")
    with open(OUT, "w") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()

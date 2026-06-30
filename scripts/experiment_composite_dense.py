"""Dense-supervision weaning experiment on the composition task (E-composite).

The unifying experiment: the s5 wall moves under dense per-step state supervision (see
experiment_dense_supervision.py). Does the SAME lever fix the *composition routing* wall on
composite_copy? I.e., if we densely supervise the holder-of-the-object (the binding leg) on the
composite, does the value (recall) leg fire too? And can we then WEAN to answer-only and keep it?

Conditions:
  - answer-only   : baseline (the routing wall).
  - dense         : supervise the holder every event; final value unsupervised (recall rides free).
  - dense->wean   : train dense, then fine-tune on answer-only; does the circuit persist?

If dense supervision makes the value leg fire, the two dissociations unify: "dense supervision
fixes routing." If the value leg stays at floor despite a correct holder trace, composition routing
is a DISTINCT wall from state-tracking.

Eval: guided free-run (events forced, holder slots + final value generated), holder/value decomp.

Example:
    .venv-train/bin/python scripts/experiment_composite_dense.py --seeds 0 1 2 3 4
"""
from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import sys
from collections import defaultdict
from datetime import datetime, timezone

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from factworld import tasks as TK
from factworld import train as T
from factworld.render import Renderer
from factworld.oracle import Oracle
from factworld.tokenizer import Tokenizer
from factworld.config import WorldConfig
from factworld.world import World


def make_world(spec):
    wc = WorldConfig(seed=spec.seed, n_entities=8, n_attributes=2,
                     value_vocab_size=spec.value_vocab_size, n_objects=spec.n_objects,
                     n_locations=6, k=spec.k)
    w = World(wc)
    return w, Renderer(), Oracle(w)


def build_composite_stream(spec, w, r, oracle, origins, L, K, rng, supervise_value=False):
    """A composite_copy_scale_v1 stream: facts + give-stream + dense holder-of-object slots + query + value.

    K = supervise the holder-of-the-object every K give-events (K=1 dense on the binding leg).
    The final value (recall leg) is NEVER supervised unless supervise_value=True (a control).
    Returns (words, hidx, vidx, obj, gold_holder, gold_value).
    """
    pool = spec.recall_pool or spec.k
    chosen = list(w.agents[:pool])
    objs = list(w.objects[:spec.n_objects_active])
    ev = [type(w.sample_easy_chain(1, "x")[0])("give", (rng.choice(objs), rng.choice(chosen))) for _ in range(L)]
    obj = rng.choice(sorted({e.args[0] for e in ev}))
    holder = oracle.easy_holder(ev, obj)
    value = origins[holder]
    facts = " ".join(r.render_fact(a, "a0", origins[a], key=f"{a}|{rng.random()}") for a in chosen)
    words = facts.split()
    hidx = []
    for i, e in enumerate(ev):
        words += ["s" + str(i), "gives" if e.kind == "give" else "moves", e.args[0], "to", e.args[1], "."]
        if (i + 1) % K == 0 or i == L - 1:
            h_after = oracle.easy_holder(ev, obj, t=i + 1)
            words += ["holder", h_after, "."]
            hidx.append(len(words) - 2)
    words += ["what", "is", "a0", "of", "the", "holder", "of", obj, "?"]
    vidx = len(words)
    words += [value, "."]
    return words, hidx, vidx, obj, holder, value


def build_docs(spec, w, r, oracle, origins, n, K, seed, supervise_value=False):
    docs = []
    for j in range(n):
        L = [4, 8, 16][j % 3]
        rng = random.Random(seed * 1000 + j)
        words, *_ = build_composite_stream(spec, w, r, oracle, origins, L, K, rng, supervise_value)
        docs.append(" ".join(words))
    return docs


def encode_docs(tok, docs):
    enc = [tok.encode(t, add_eos=True) for t in docs]
    enc.sort(key=len)
    return enc


def eval_value_given_trace(model, tok, w, origins, spec, oracle, r, n, L, device="cuda", max_slot=4):
    """The routing test: facts+events+CORRECT holder trace all TEACHER-FORCED; only the final
    VALUE is generated. This isolates 'given the right state, can the model route it into recall?'
    from 'can the model track state'. If this stays at floor under dense training, routing is a
    DISTINCT wall from state-tracking."""
    import torch
    val_ids = {tok.token_to_id[v] for v in w.value_vocab}
    dot = tok.token_to_id["."]
    model.eval()
    value_ok = 0
    for j in range(n):
        # build a dense stream so the holder trace is in the prompt; force everything except the value
        words, hidx, vidx, obj, gold_h, gold_v = build_composite_stream(
            spec, w, r, oracle, origins, L, 1, random.Random(900 + L + j))
        # teacher-force the entire stream up to the value slot
        forced = words[:vidx]
        ids = tok.encode(" ".join(forced))
        gen = None
        with torch.no_grad():
            for _ in range(max_slot):
                with torch.autocast(device, dtype=torch.bfloat16):
                    nx = int(model(torch.tensor([ids], device=device))[0, -1].float().argmax())
                ids.append(nx)
                if gen is None and nx in val_ids:
                    gen = nx
                if nx == dot:
                    break
        value_ok += int(gen == tok.token_to_id[gold_v])
    return value_ok / n


def eval_free_run(model, tok, w, origins, spec, oracle, r, n, L, K, device="cuda", max_slot=4):
    """Guided free-run on composite: facts+events forced, holder slots + final value GENERATED."""
    import torch
    ag_ids = {tok.token_to_id[g] for g in w.agents}
    val_ids = {tok.token_to_id[v] for v in w.value_vocab}
    dot = tok.token_to_id["."]
    model.eval()
    holder_ok = value_ok = 0
    for j in range(n):
        words, hidx, vidx, obj, gold_h, gold_v = build_composite_stream(
            spec, w, r, oracle, origins, L, K, random.Random(900 + L + j))
        slots = set(hidx) | {vidx}
        ids, i, last_holder = [], 0, None
        with torch.no_grad():
            while i < len(words):
                if i in slots:
                    gen = None
                    for _ in range(max_slot):
                        with torch.autocast(device, dtype=torch.bfloat16):
                            nx = int(model(torch.tensor([ids], device=device))[0, -1].float().argmax())
                        ids.append(nx)
                        ts = ag_ids if i in hidx else val_ids
                        if gen is None and nx in ts:
                            gen = nx
                        if nx == dot:
                            break
                    if i in hidx:
                        last_holder = gen
                    else:
                        value_ok += int(gen == tok.token_to_id[gold_v])
                    i += 1
                    while i < len(words) and words[i] != ".":
                        i += 1
                    i += 1
                else:
                    ids += tok.encode(words[i]); i += 1
        holder_ok += int(last_holder == tok.token_to_id[gold_h])
    return holder_ok / n, value_ok / n


def run_condition(spec, cond, seed, *, steps, wean_steps, batch, d_model, n_layers, train_n, device):
    """Train one (cond, seed). cond in {answer_only, dense, dense_wean}."""
    w, r, oracle = make_world(spec)
    origins = TK._fixed_origins(spec, w)
    tok = Tokenizer.build([w], r)

    if cond == "answer_only":
        docs = build_docs(spec, w, r, oracle, origins, train_n, 10 ** 9, seed)
        run = T.run("gdp_hybrid", tok, encode_docs(tok, docs), [], steps=steps, batch=batch,
                    d_model=d_model, n_layers=n_layers, d_ff=4 * d_model, seed=seed,
                    return_model=True, device=device)
        return run["model"], tok, w, r, oracle, origins

    if cond == "dense":
        docs = build_docs(spec, w, r, oracle, origins, train_n, 1, seed)
        run = T.run("gdp_hybrid", tok, encode_docs(tok, docs), [], steps=steps, batch=batch,
                    d_model=d_model, n_layers=n_layers, d_ff=4 * d_model, seed=seed,
                    return_model=True, device=device)
        return run["model"], tok, w, r, oracle, origins

    if cond == "dense_wean":
        # Phase 1: dense; Phase 2: continue training on answer-only (same total-ish budget split).
        docs_dense = build_docs(spec, w, r, oracle, origins, train_n, 1, seed)
        run = T.run("gdp_hybrid", tok, encode_docs(tok, docs_dense), [], steps=steps, batch=batch,
                    d_model=d_model, n_layers=n_layers, d_ff=4 * d_model, seed=seed,
                    return_model=True, device=device)
        # phase 2: answer-only fine-tune from the dense checkpoint
        docs_ao = build_docs(spec, w, r, oracle, origins, train_n, 10 ** 9, seed)
        run2 = T.run("gdp_hybrid", tok, encode_docs(tok, docs_ao), [], steps=wean_steps, batch=batch,
                     d_model=d_model, n_layers=n_layers, d_ff=4 * d_model, seed=seed + 777,
                     return_model=True, device=device)
        return run2["model"], tok, w, r, oracle, origins

    raise ValueError(cond)


def main():
    ap = argparse.ArgumentParser(description="Dense-supervision weaning on the composition task.")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--conditions", nargs="+", default=["answer_only", "dense", "dense_wean"])
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--wean_steps", type=int, default=4000, help="Answer-only fine-tune steps for dense_wean.")
    ap.add_argument("--d_model", type=int, default=256)
    ap.add_argument("--n_layers", type=int, default=4)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--train_n", type=int, default=8000)
    ap.add_argument("--eval_n", type=int, default=100)
    ap.add_argument("--eval_lengths", type=int, nargs="+", default=[16, 64])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out_prefix", default=None)
    a = ap.parse_args()

    spec = TK.CANONICAL["composite_copy_scale_v1"]
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    from pathlib import Path
    prefix = Path(a.out_prefix or f"results/composite_dense_{ts}")
    prefix.parent.mkdir(parents=True, exist_ok=True)
    jsonl = Path(f"{prefix}.jsonl"); md = Path(f"{prefix}.md")

    print(f"=== composite dense-weaning -> {jsonl} ===", flush=True)
    agg = defaultdict(lambda: defaultdict(list))
    for cond in a.conditions:
        for seed in a.seeds:
            print(f"\n--- {cond} seed={seed} ---", flush=True)
            model, tok, w, r, oracle, origins = run_condition(
                spec, cond, seed, steps=a.steps, wean_steps=a.wean_steps, batch=a.batch,
                d_model=a.d_model, n_layers=a.n_layers, train_n=a.train_n, device=a.device)
            row = {"condition": cond, "seed": seed}
            for L in a.eval_lengths:
                # Three metrics: (1) free-run holder (did it learn to track state?)
                #                (2) free-run value = END-TO-END (self-generated holder -> value)
                #                (3) value GIVEN the correct trace = the routing ceiling
                fh, fv_freerun = eval_free_run(model, tok, w, origins, spec, oracle, r, a.eval_n, L, 1, device=a.device)
                v_gt = eval_value_given_trace(model, tok, w, origins, spec, oracle, r, a.eval_n, L, device=a.device)
                row[f"L{L}_holder_freerun"] = fh
                row[f"L{L}_value_freerun"] = fv_freerun
                row[f"L{L}_value_given_trace"] = v_gt
                agg[cond][L].append(fv_freerun)   # headline = end-to-end free-run value
                print(f"    L{L}: holder(free-run)={fh:.2f}  value(free-run/end2end)={fv_freerun:.2f}  "
                      f"value(given correct trace)={v_gt:.2f}", flush=True)
            with jsonl.open("a") as f:
                f.write(json.dumps(row) + "\n")
            import torch; del model; torch.cuda.empty_cache()

    lines = ["# Dense-supervision weaning on the composition task (composite_copy_scale_v1)", "",
             f"`scripts/experiment_composite_dense.py`. gdp_hybrid d{a.d_model}x{a.n_layers}, {a.steps} steps "
             f"(+{a.wean_steps} wean), seeds {a.seeds}. Guided free-run: facts+events forced, holder slots + "
             f"final value GENERATED. value = the recall leg (does it fire?). Floor = 0.20.", "",
             "| condition | " + " | ".join(f"value @L{L}" for L in a.eval_lengths) + " |",
             "|---|" + "---|" * len(a.eval_lengths)]
    for cond in a.conditions:
        cells = []
        for L in a.eval_lengths:
            xs = agg[cond][L]
            cells.append(f"{statistics.mean(xs):.2f}±{statistics.pstdev(xs):.2f}" if xs else "-")
        lines.append(f"| {cond} | " + " | ".join(cells) + " |")
    lines += ["", "_value = END-TO-END free-run accuracy (self-generated holder -> value), the headline metric. "
              "holder(free-run) and value(given correct trace) are in the JSONL: the former is the state-tracking leg, "
              "the latter the routing ceiling. If dense value > answer-only value, dense supervision fixes composition "
              "(by fixing the holder leg the routing ceiling is already ~1.0)._"]
    md.write_text("\n".join(lines))
    print(f"\n=== wrote {md} ===", flush=True)


if __name__ == "__main__":
    main()

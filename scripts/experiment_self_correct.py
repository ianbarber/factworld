"""Self-correction / iterative-refinement test-time probe (E-refine).

The strong test of 'test-time compute does not move the wall'. We already showed sampling-based
self-consistency (majority vote to 30) does not help. This probes the STRONGER form: iterative
refinement — generate a holder, then re-prompt the model to verify it against the events and
regenerate, for N correction rounds. If iterative self-correction also fails on seeds that formed a
partial circuit, the null ('test-time reasoning does not help') upgrades from weak to strong.

Run on a K=2 (partial-circuit) model from the dense sweep, at L64 (the bimodal length).

Example:
    .venv-train/bin/python scripts/experiment_self_correct.py --K 2 --seeds 0 1 2
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts"))
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from factworld import tasks as TK
from factworld.render import Renderer
import importlib.util

# reuse the dense-supervision training + eval scaffolding
_spec = importlib.util.spec_from_file_location("eds", os.path.join(REPO, "scripts", "experiment_dense_supervision.py"))
eds = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(eds)


def gen_slot(model, tok, ids, device, valid_ids, dot, max_slot=4, temp=0.0):
    """Generate one slot (holder or value) greedily (temp=0) or sampled (temp>0). Return (gen_id, new_ids)."""
    import torch
    gen = None
    new_ids = list(ids)
    gen_g = torch.Generator(device=device).manual_seed(0) if temp > 0 else None
    for _ in range(max_slot):
        with torch.autocast(device, dtype=torch.bfloat16):
            logits = model(torch.tensor([new_ids], device=device))[0, -1].float()
        if temp > 0:
            nx = int(torch.multinomial(torch.softmax(logits / temp, dim=-1), 1, generator=gen_g).item())
        else:
            nx = int(logits.argmax())
        new_ids.append(nx)
        if gen is None and nx in valid_ids:
            gen = nx
        if nx == dot:
            break
    return gen, new_ids


def self_correct_eval(model, tok, w, exs, device="cuda", rounds=3):
    """Iterative refinement: generate holder, then re-prompt 'is <h> the holder? if wrong, what is it?'
    for `rounds` correction cycles. Score the final holder + value. Compare to round-0 (no correction)."""
    import torch
    ag_ids = {tok.token_to_id[g] for g in w.agents}
    val_ids = {tok.token_to_id[v] for v in w.value_vocab}
    dot = tok.token_to_id["."]
    model.eval()
    # track accuracy at each round
    holder_acc_by_round = {r: 0 for r in range(rounds + 1)}
    value_acc_by_round = {r: 0 for r in range(rounds + 1)}
    n = len(exs)
    with torch.no_grad():
        for words, hidx, vidx, role, true_fh, true_val in exs:
            # build the forced prefix up to the FIRST holder slot (the final one, hidx[-1])
            final_h_idx = hidx[-1]
            # forced = everything before the final holder slot
            forced = []
            i = 0
            while i < final_h_idx:
                forced += tok.encode(words[i]); i += 1
            # round 0: generate the holder
            cur_holder, ids = gen_slot(model, tok, forced, device, ag_ids, dot)
            for r in range(rounds + 1):
                holder_acc_by_round[r] += int(cur_holder == tok.token_to_id[true_fh])
                # generate the value from THIS holder, to measure end-to-end at this round
                # reconstruct: forced + "holder <cur> . " + query, then generate value
                val_prefix = forced + tok.encode(" ".join(["holder", tok.id_to_token.get(cur_holder, "?"), "."]))
                # append everything from after the holder slot up to the value slot
                vi = final_h_idx
                # skip the true holder tokens
                while vi < len(words) and words[vi] != ".":
                    vi += 1
                vi += 1
                while vi < vidx:
                    val_prefix += tok.encode(words[vi]); vi += 1
                vgen, _ = gen_slot(model, tok, val_prefix, device, val_ids, dot)
                value_acc_by_round[r] += int(vgen == tok.token_to_id[true_val])
                if r == rounds:
                    break
                # correction round: re-prompt "the holder was <cur>. check the events. the holder is" and regenerate
                correct_prefix = forced + tok.encode(" ".join(["holder", tok.id_to_token.get(cur_holder, "?"), ".", "check", "the", "holder", "is"]))
                new_holder, _ = gen_slot(model, tok, correct_prefix, device, ag_ids, dot, temp=0.0)
                if new_holder is not None:
                    cur_holder = new_holder
    return ({r: holder_acc_by_round[r] / n for r in range(rounds + 1)},
            {r: value_acc_by_round[r] / n for r in range(rounds + 1)})


def main():
    ap = argparse.ArgumentParser(description="Self-correction test-time probe.")
    ap.add_argument("--K", type=int, default=2, help="Supervision stride to train at (2 = partial circuit).")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--rounds", type=int, default=3, help="Self-correction rounds.")
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--eval_L", type=int, default=64)
    ap.add_argument("--eval_n", type=int, default=100)
    ap.add_argument("--out_prefix", default=None)
    a = ap.parse_args()

    spec = TK.CANONICAL["composite_copy_scale_v1"].scaled(k=5)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    from pathlib import Path
    prefix = Path(a.out_prefix or f"results/self_correct_K{a.K}_{ts}")
    prefix.parent.mkdir(parents=True, exist_ok=True)
    jsonl = Path(f"{prefix}.jsonl"); md = Path(f"{prefix}.md")

    print(f"=== self-correction probe (K={a.K}, {a.rounds} rounds) -> {jsonl} ===", flush=True)
    rows = []
    for seed in a.seeds:
        print(f"\n--- K={a.K} seed={seed} ---", flush=True)
        model, tok, w, r, oracle, origins = eds.run_K(spec, a.K, seed, steps=a.steps, batch=32,
                                                       d_model=256, n_layers=4, train_n=8000,
                                                       device="cuda", arch="gdp_hybrid")
        exs = eds.build_eval(spec, w, r, oracle, origins, a.eval_L, a.K, a.eval_n)
        # greedy baseline (rounds=0 essentially)
        h_by_r, v_by_r = self_correct_eval(model, tok, w, exs, device="cuda", rounds=a.rounds)
        row = {"K": a.K, "seed": seed, "L": a.eval_L,
               "holder_by_round": h_by_r, "value_by_round": v_by_r}
        with jsonl.open("a") as f:
            f.write(json.dumps(row, default=str) + "\n")
        rows.append(row)
        print(f"    holder by round: {h_by_r}", flush=True)
        print(f"    value  by round: {v_by_r}", flush=True)
        import torch; del model; torch.cuda.empty_cache()

    # markdown: mean across seeds by round
    import statistics
    lines = [f"# Self-correction test-time probe (K={a.K}, {a.eval_L})", "",
             f"`scripts/experiment_self_correct.py`. gdp_hybrid trained at K={a.K} (partial circuit), "
             f"evaluated at L{a.eval_L}. Round 0 = no correction; each round re-prompts the model to "
             f"'check' and regenerate the holder. If accuracy is flat across rounds, iterative "
             f"self-correction does not help (strong null for test-time compute).", "",
             "| round | holder acc | value acc |",
             "|---|---|---|"]
    for rnd in range(a.rounds + 1):
        hs = [float(r["holder_by_round"].get(str(rnd), r["holder_by_round"].get(rnd, 0))) for r in rows]
        vs = [float(r["value_by_round"].get(str(rnd), r["value_by_round"].get(rnd, 0))) for r in rows]
        lines.append(f"| {rnd} | {statistics.mean(hs):.2f} | {statistics.mean(vs):.2f} |")
    md.write_text("\n".join(lines))
    print(f"\n=== wrote {md} ===", flush=True)


if __name__ == "__main__":
    main()

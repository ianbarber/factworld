"""Staged curriculum for s5_chain — the sub-circuits first, then the composition.

QUESTION
    Does s5_chain form locally when its two components are trained before the composition,
    at the budget the composite that DID converge was given?

    Every s5_chain run to date trained the composition cold: 8k documents, 8k steps, batch
    32, d320x4 (~32 epochs over 8k documents). The composite task that converged locally used
    80k documents, 25k steps, batch 128, d768x8, AND a staged curriculum
    (scripts/experiment_curriculum_staged.py). s5_chain appears in no curriculum script, so
    "s5_chain does not form" has never been separated from "s5_chain was not given the recipe
    that made anything else form".

PROTOCOL
    One world, one tokenizer, one model, carried through three stages (each stage continues
    from the previous model's weights, as in experiment_curriculum_staged.py):

      stage 1  chain           serial dereference over a STATED pointer map, no events
      stage 2  chain + s5_chain depth 1   dereference over a map the events have moved
      stage 3  + s5_chain depth 2         the composition

    Earlier arms stay in the mix at decaying weight so the sub-circuits are not overwritten.
    Every stage is evaluated on the depth-1 and depth-2 s5_chain held-out lengths, so the
    stage at which a component stops holding is visible rather than inferred.

DECISION RULE (pre-registered)
    Read every cell against its own OPERATIVE floor — the max over the registered shallow
    adversaries (factworld.validity), recomputed from the exact scored items — and report
    per-seed values, since this family is bimodal. Naming one adversary understates the floor:
    at k=6/depth=2 the operative floor is 1/(k-1) = 0.200 at both lengths, above the
    initial-map chase at either. The stage-1 chain arm has no event stream, so no chase row is
    registered for it at all; its floor is chance.

      depth-2 clears floor + 0.25 on >= 3 of 8 seeds at the trained length
          -> s5_chain forms locally under a curriculum. The published local null was a budget
             and curriculum artifact. Promote to a scored local arm and re-run the k/depth
             grid under this recipe.
      depth-1 clears but depth-2 does not
          -> tracking + single readout forms; composing the second hop on top does not. That
             is the composition result, and it is only claimable with the depth-1 arm as the
             positive control in the same table.
      neither clears at stage 3, but stage 1 chain accuracy is high
          -> the dereference circuit exists and does not survive contact with the event
             stream. Next probe is supervision shape (guided free-run), not scale.
      stage 1 chain does not clear
          -> the harness is not training at this width/depth at all; nothing downstream is
             readable and no s5_chain claim of any kind can be made from this run.

COST
    1.3-2.8 GPU-h per (arch, seed) at the default d768x8 / 25k steps / batch 128 / 80k docs on
    a single 5090 (scripts/remeasure_v2_issue11.sh:60 — 9 runs of that recipe in 12-25 h).
    Pure compute scaling off the measured 6.6 min/run at d768x8 / 8k steps / batch 32
    (logs/s5_chain_local_d768_20260718.log) lands at the bottom of that range. 2 archs x 3
    seeds ~ 8-17 GPU-h; 2 archs x 8 seeds ~ 21-45 GPU-h. On top of that sits the evaluation:
    every stage scores all three arms at every eval length, 18 arm-evals per run against 2 for
    a plain sweep run. Run the 3-seed version first and only extend to 8 seeds if any cell
    clears its floor.

Example (smoke test, minutes):
    .venv-train/bin/python scripts/experiment_s5_chain_curriculum.py \
        --seeds 0 --steps 600 --d_model 128 --n_layers 2 --batch 8 --train_n 300 --eval_n 20

Full run:
    .venv-train/bin/python scripts/experiment_s5_chain_curriculum.py \
        --archs gdp_hybrid,fprm --seeds 0 1 2 --steps 25000 --batch 128 \
        --d_model 768 --n_layers 8 --train_n 80000
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
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from factworld import tasks as TK  # noqa: E402
from factworld.backends import LocalBackend  # noqa: E402
from factworld.runner import evaluate_task  # noqa: E402
from factworld.validity import operative_floor  # noqa: E402
from sweep import build_docs, cell_floors, held_out_loss, trace_budget  # noqa: E402

# Stage weights: (arm name -> share of that stage's training documents).
DEFAULT_SCHEDULE = (
    ("stage1", 0.30, {"chain": 1.0}),
    ("stage2", 0.30, {"chain": 0.30, "d1": 0.70}),
    ("stage3", 0.40, {"chain": 0.15, "d1": 0.35, "d2": 0.50}),
)


def curriculum_specs(k, train_lengths, eval_lengths, event_trace):
    """The three training arms, all on ONE world (same k / vocab / seed -> same tokenizer).

    ``chain`` reinterprets ``length`` as the chain DEPTH, so its lengths are depths 1-2: the
    dereference the s5_chain query performs, over a map no event has touched.

    No arm is renamed. ``spec.name`` keys the RNG stream (and, for chain, whether the query
    carries the ``(N hops)`` annotation), so renaming would both desynchronise these arms from
    the s5_chain sweep cells they are meant to be compared against and silently change the
    chain arm's query grammar. Knobs are overridden the way scripts/sweep.py's --k /
    --chain_depth flags override them.
    """
    base = TK.CANONICAL["s5_chain_local_v2"].scaled(
        k=k, distinct_path=True, worked_trace=True, event_trace=event_trace,
        train_lengths=train_lengths, eval_lengths=eval_lengths)
    return {
        "chain": TK.CANONICAL["chain_v2"].scaled(
            k=k, value_vocab_size=base.value_vocab_size, n_objects=base.n_objects,
            worked_trace=True, train_lengths=(1, 2), eval_lengths=(2,)),
        "d1": base.scaled(chain_depth=1),
        "d2": base.scaled(chain_depth=2),
    }


def stage_docs(specs, weights, train_n, use_trace):
    """Training documents for one stage: ``train_n`` documents split by the stage weights."""
    docs = []
    for arm, share in sorted(weights.items()):
        n = int(round(train_n * share))
        if n <= 0:
            continue
        docs += build_docs(TK.generate(specs[arm], "train", n=n), use_trace, False)
    return docs


def eval_arm(model, tok, spec, arch, device, eval_n, use_trace):
    """Score one arm at every eval length, with its floors and held-out loss."""
    backend = LocalBackend([], arch=arch, model=model, tokenizer=tok, device=device)
    out = {}
    for L in spec.eval_lengths:
        if use_trace:
            probe = TK.generate(spec, "test", n=1, length=L)[0]
            max_new = trace_budget(spec, L, len(probe.meta.get("trace", "").split()))
        else:
            max_new = None
        res = evaluate_task(backend, spec, split="test", n=eval_n, length=L,
                            max_new_tokens=max_new, stop_at="<eos>" if use_trace else ".")
        overall = res["metrics"]["last_n"]["overall"] if use_trace else res["overall"]
        heldout = build_docs(TK.generate(spec, "test", n=min(eval_n, 200), length=L),
                             use_trace, False)
        out[str(L)] = {
            "overall": overall,
            "floors": cell_floors(spec, L, eval_n),
            "heldout_loss": held_out_loss(model, tok, heldout, device),
            "preds_sample": [{"gold": g, "pred": p, "correct": ok}
                             for _pr, g, p, ok in res["examples"][:10]],
        }
    return out


def run_one(arch, seed, specs, schedule, *, steps, batch, d_model, n_layers, train_n, eval_n,
            device, use_trace, loss_log_interval):
    """Train one (arch, seed) through every stage; return the per-stage records."""
    import torch

    from factworld import train as T
    from factworld.tokenizer import Tokenizer

    world, renderer = TK.build_world(specs["d2"])
    tok = Tokenizer.build([world], renderer)
    model, records = None, []
    for name, share, weights in schedule:
        stage_steps = max(1, int(round(steps * share)))
        texts = stage_docs(specs, weights, train_n, use_trace)
        docs = [tok.encode(t, add_eos=True)[:1280] for t in texts]
        docs.sort(key=len)
        print(f"  -- {name}: {stage_steps} steps, {len(docs)} docs, mix={weights}", flush=True)
        run = T.run(arch, tok, docs, [], steps=stage_steps, batch=batch, d_model=d_model,
                    n_layers=n_layers, d_ff=4 * d_model, seed=seed, return_model=True,
                    device=device, model=model, loss_log_interval=loss_log_interval)
        model = run["model"]
        rec = {"stage": name, "steps": stage_steps, "n_docs": len(docs),
               "epochs": round(stage_steps * batch / max(1, len(docs)), 2),
               "mix": weights, "final_loss": run["final_loss"],
               "loss_curve": run.get("loss_curve", []), "eval": {}}
        for arm in ("chain", "d1", "d2"):
            rec["eval"][arm] = eval_arm(model, tok, specs[arm], arch, device, eval_n, use_trace)
        for arm, cells in rec["eval"].items():
            # operative floor = max over the registered adversaries; naming one row understates
            # it on the low-k cells, and on the event-free `chain` arm the initial-map chase is
            # the oracle (1.000) rather than a floor, so it is not reported there at all.
            print(f"     {arm}: " + "  ".join(
                f"L{L}={c['overall']:.2f}(floor "
                f"{float('nan') if operative_floor(c['floors']) is None else operative_floor(c['floors']):.2f})"
                for L, c in cells.items()), flush=True)
        records.append(rec)
    del model
    torch.cuda.empty_cache()
    return records


def write_markdown(rows, cfg, path):
    lines = [f"# s5_chain staged curriculum — {cfg['archs']}", "",
             f"d_model={cfg['d_model']} n_layers={cfg['n_layers']} steps={cfg['steps']} "
             f"batch={cfg['batch']} train_n={cfg['train_n']} seeds={cfg['seeds']}", ""]
    for arm, label in (("chain", "chain (dereference only)"),
                       ("d1", "s5_chain depth 1"), ("d2", "s5_chain depth 2")):
        lines.append(f"## {label}")
        lengths = sorted({L for r in rows for st in r["stages"] for L in st["eval"][arm]},
                         key=int)
        lines.append("| arch | stage | " + " | ".join(f"L{L} per-seed" for L in lengths) + " |")
        lines.append("|" + "---|" * (len(lengths) + 2))
        stages = [st["stage"] for st in rows[0]["stages"]]
        archs = sorted({r["arch"] for r in rows})
        for arch in archs:
            for si, stage in enumerate(stages):
                cells = []
                for L in lengths:
                    vals = [r["stages"][si]["eval"][arm][L]["overall"]
                            for r in rows if r["arch"] == arch and L in r["stages"][si]["eval"][arm]]
                    cells.append(" ".join(f"{v:.2f}" for v in vals) if vals else "—")
                lines.append(f"| {arch} | {stage} | " + " | ".join(cells) + " |")
        floors, named = {}, {}
        for r in rows:
            for L, c in r["stages"][-1]["eval"][arm].items():
                floors.setdefault(L, operative_floor(c["floors"]))
                named.setdefault(L, max(c["floors"], key=c["floors"].get) if c["floors"] else "—")
        cells = [f"**{floors[L]:.3f}** ({named[L]})" if floors.get(L) is not None else "—"
                 for L in lengths]
        lines.append("| _operative floor_ | — | " + " | ".join(cells) + " |")
        lines.append("")
    lines.append("_Per-seed values in seed order (bimodal family: a mean hides a converged "
                 "seed). The operative floor is the largest shallow adversary for that cell, "
                 "recomputed from the exact scored items; the named row varies with k, depth "
                 "and length. The chain arm has no event stream, so the initial-map chase is "
                 "its oracle rather than an adversary and is not registered there._")
    path.write_text("\n".join(lines))


def main():
    ap = argparse.ArgumentParser(description="Staged curriculum for s5_chain.")
    ap.add_argument("--archs", default="gdp_hybrid,fprm")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--steps", type=int, default=25000, help="Total steps across all stages.")
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--d_model", type=int, default=768)
    ap.add_argument("--n_layers", type=int, default=8)
    ap.add_argument("--train_n", type=int, default=80000, help="Documents per stage.")
    ap.add_argument("--eval_n", type=int, default=200)
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--train_lengths", type=int, nargs="+", default=[2, 4])
    ap.add_argument("--eval_lengths", type=int, nargs="+", default=[4, 8])
    ap.add_argument("--no_event_trace", action="store_true",
                    help="Path-only traces instead of per-event map checkpoints.")
    ap.add_argument("--loss_log_interval", type=int, default=200)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out_prefix", default=None)
    a = ap.parse_args()

    archs = [x.strip() for x in a.archs.split(",")]
    specs = curriculum_specs(a.k, tuple(a.train_lengths), tuple(a.eval_lengths),
                             not a.no_event_trace)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    prefix = Path(a.out_prefix or f"results/s5_chain_curriculum_{ts}")
    prefix.parent.mkdir(parents=True, exist_ok=True)
    log_path, md_path = Path(f"{prefix}.jsonl"), Path(f"{prefix}.md")
    cfg = {"archs": archs, "seeds": a.seeds, "steps": a.steps, "batch": a.batch,
           "d_model": a.d_model, "n_layers": a.n_layers, "train_n": a.train_n,
           "eval_n": a.eval_n, "k": a.k, "schedule": [(n, s, w) for n, s, w in DEFAULT_SCHEDULE]}

    rows = []
    total = len(archs) * len(a.seeds)
    print(f"=== s5_chain staged curriculum: {total} runs -> {log_path} ===", flush=True)
    for arch in archs:
        for seed in a.seeds:
            print(f"\n--- {arch} seed {seed} ---", flush=True)
            try:
                stages = run_one(arch, seed, specs, DEFAULT_SCHEDULE, steps=a.steps,
                                 batch=a.batch, d_model=a.d_model, n_layers=a.n_layers,
                                 train_n=a.train_n, eval_n=a.eval_n, device=a.device,
                                 use_trace=True, loss_log_interval=a.loss_log_interval)
            except Exception as e:  # noqa: BLE001
                import traceback; traceback.print_exc()
                with log_path.open("a") as f:
                    f.write(json.dumps({"arch": arch, "seed": seed, "error": str(e)}) + "\n")
                continue
            row = {"arch": arch, "seed": seed, **cfg, "stages": stages}
            rows.append(row)
            with log_path.open("a") as f:
                f.write(json.dumps(row) + "\n")
            write_markdown(rows, cfg, md_path)
    if rows:
        write_markdown(rows, cfg, md_path)
    print(f"\n=== done: {md_path} ===", flush=True)


if __name__ == "__main__":
    main()

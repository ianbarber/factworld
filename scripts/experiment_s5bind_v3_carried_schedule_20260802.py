"""IS THE TRANSFORMER NULL AN OPTIMISER ARTIFACT? Retrain it with the schedule carried across
the three stages and read the same cells.

THE DEFECT. ``experiment_s5bind_v3_three_cell_local_20260731.py`` runs the curriculum as three
calls to ``factworld.train.run``, one per stage, carrying the MODEL and nothing else. Each call
builds a fresh ``AdamW`` and restarts warmup+cosine from step 0, so a nominally 25,000-step run is
three 8,750/7,500/8,750-step runs whose learning rate goes 0 -> 1e-3 -> 0 three times and whose
Adam moments are discarded twice. That is a training-recipe property, not an architecture one, and
the arm it is read on is an ARCHITECTURE comparison.

WHY IT IS A CANDIDATE EXPLANATION FOR THIS PARTICULAR NULL. The teacher-forced probe
(``probe_s5bind_v3_teacher_forced_slots_20260801.py``) reads the transformer at 0.240-0.527 on
moving slots composed and 0.575-0.669 state, against gdp_hybrid 1.000 and fprm 0.996-1.000 at the
same width, depth, documents and step count; and its three seeds read 0.000 / 0.590 / 1.000 on the
SAME bind@31 cell. A capacity claim does not produce a 0/0.59/1 spread on one cell at one width;
an optimisation one does.

WHAT CHANGES, AND IT IS ONE THING. The three stages share ONE optimizer and ONE warmup+cosine
schedule over the global 25,000 steps (``train.run(opt=..., sched_step0=..., sched_total=...)``,
whose defaults reproduce the per-stage schedule exactly). Documents, mixes, step shares, seeds,
width, depth, batch, lr, eval grid, floors and both reads are the registered ones, and this script
patches ONLY ``run_one`` — the grid, the floors, the rule and the report are the registered
module's own functions, so the two runs are read by identical code.

WHAT IT CANNOT SETTLE. If the null survives, the null is an architecture result at this
(width, depth, budget, recipe) and nothing here says it is one anywhere else. If the null does not
survive, then EVERY arm's numbers were taken under the restarting schedule, and the comparison
that stands is all three architectures under one schedule — not this arm against the old ones.

Usage (the registered operating point):
    .venv-train/bin/python scripts/experiment_s5bind_v3_carried_schedule_20260802.py \
        --archs transformer --seeds 0 1 2 --steps 25000 --batch 16 \
        --d_model 768 --n_layers 8 --n_heads 6 --train_n 80000 --eval_n 1000 \
        --guided_batch 32 --floors results/s5bind_v3_three_cell_depthmatched_20260801.json \
        --out_prefix results/s5bind_v3_carried_schedule_20260802
"""
from __future__ import annotations

import os
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts"))
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import protocol_s5bind_v3_three_cell_20260731 as P                         # noqa: E402
import experiment_s5bind_v3_three_cell_local_20260731 as E                 # noqa: E402


def run_one_carried(arch, seed, specs, tok, world, grid, *, steps, batch, d_model, n_layers,
                    n_heads, lr, train_n, eval_n, guided_n, guided_lengths, device, fmt,
                    loss_log_interval, guided_batch=128, ckpt_dir=None):
    """``E.run_one`` with the optimizer and the LR schedule carried across the three stages.

    Line for line the registered ``run_one`` except for ``opt``/``sched_step0``/``sched_total``:
    same SCHEDULE, same ``stage_documents`` (cached and deterministic, so every arm sees the same
    documents in the same order), same per-stage checkpoint write, same evaluation.
    """
    import torch
    from factworld import train as T

    model, opt, stages, done = None, None, [], 0
    for si, (name, share, weights) in enumerate(E.SCHEDULE):
        last = si == len(E.SCHEDULE) - 1
        stage_steps = max(1, int(round(steps * share)))
        docs, plens = E.stage_documents(specs, weights, train_n, tok, fmt)
        t0 = time.time()
        run = T.run(arch, tok, docs, [], steps=stage_steps, batch=batch, d_model=d_model,
                    n_layers=n_layers, n_heads=n_heads, d_ff=4 * d_model, lr=lr, seed=seed,
                    return_model=True, device=device, model=model, use_short_conv=True,
                    loss_log_interval=loss_log_interval, prompt_lens=plens,
                    opt=opt, sched_step0=done, sched_total=steps, return_opt=True)
        model, opt = run["model"], run["opt"]
        done += stage_steps
        print(f"  -- {name}: {stage_steps} steps (global {done - stage_steps}-{done} of {steps}), "
              f"{len(docs)} docs, loss={run['final_loss']:.4f} [{time.time() - t0:.0f}s]",
              flush=True)
        if ckpt_dir:
            E.save_checkpoint(model, E.checkpoint_path(ckpt_dir, arch, seed), arch=arch,
                              seed=seed, stage=name,
                              build={"d_model": d_model, "n_layers": n_layers,
                                     "n_heads": n_heads, "d_ff": 4 * d_model,
                                     "use_short_conv": True, "vocab_size": tok.vocab_size},
                              provenance={"steps": stage_steps, "n_docs": len(docs),
                                          "mix": weights, "final_loss": run["final_loss"],
                                          "lr": lr, "batch": batch, "fmt": fmt,
                                          "train_n": train_n,
                                          "train_lengths": list(P.TRAIN_LENGTHS),
                                          "schedule": "carried",
                                          "sched_step0": done - stage_steps,
                                          "sched_total": steps})
        if last:
            ev, gv = E.evaluate_all(model, arch, specs, tok, world, grid, eval_n=eval_n,
                                    guided_n=guided_n, guided_lengths=guided_lengths,
                                    device=device, guided_batch=guided_batch)
        else:
            ev, gv = E.evaluate_all(
                model, arch, specs, tok, world,
                {c: [P.CONTROL_LENGTH, P.registered_lengths(c)[0]] for c in grid},
                eval_n=200, guided_n=0, guided_lengths={}, device=device)
        stages.append({"stage": name, "steps": stage_steps, "n_docs": len(docs),
                       "mix": weights, "final_loss": run["final_loss"],
                       "loss_curve": [(int(s), float(v)) for s, v in run.get("loss_curve", [])],
                       "train_s": round(time.time() - t0), "eval": ev, "guided": gv,
                       "sched_step0": done - stage_steps, "sched_total": steps})
    del model, opt
    torch.cuda.empty_cache()
    return stages


if __name__ == "__main__":
    E.run_one = run_one_carried
    E.main()

"""IS THE NULL A MISSING RULE OR A COMPOUNDING ERROR? One forward pass per item says which.

The GUIDED read is FREE-RUNNING on the checkpoints: the events are teacher-forced but every slot
the model writes is fed back, so one wrong slot is carried into every later checkpoint. A cell at
chance under that read has two completely different explanations and the registered numbers cannot
tell them apart:

    NOT FORMED     the model never learned the per-event update, so its next-slot prediction is
                   wrong even when the whole true history is in front of it.
    NOT SURVIVED   the model has the update but not the closed loop: given the true prefix it
                   predicts the next slot, and free-running its own errors compound away from it.

This probe reads the SAME slots under TEACHER FORCING — the gold interleaved document, one forward
pass, argmax at each slot position against the gold token there — so the two readings separate. It
is a DIAGNOSTIC and no verdict reads it: teacher-forced slot accuracy is not a score on the task,
because the true history is exactly what the task withholds.

It costs one forward per item against the free-running read's (k + m) L sequential rounds, so the
whole grid is minutes rather than hours.

Usage:
    .venv-train/bin/python scripts/probe_s5bind_v3_teacher_forced_slots_20260801.py \
        --ckpt_dir results/s5bind_v3_three_cell_depthmatched_20260801_ckpt \
        --out results/s5bind_v3_teacher_forced_slots_20260801.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts"))
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from factworld import tasks as TK                                          # noqa: E402
from factworld.render import Renderer                                      # noqa: E402
from factworld.tokenizer import Tokenizer                                  # noqa: E402
import protocol_s5bind_v3_three_cell_20260731 as P                         # noqa: E402
import experiment_s5bind_v3_three_cell_local_20260731 as E                 # noqa: E402

CELLS = (("state", 17), ("state", 23), ("state", 34), ("state", 80),
         ("bind", 31), ("bind", 41), ("bind", 62),
         ("composed", 48), ("composed", 64), ("composed", 96))
ARCH_ORDER = ("gdp_hybrid", "fprm", "transformer")


def teacher_forced_slots(model, tok, spec, length, n, device, batch=16):
    """``{per_slot, trace, n, moving_slots}`` — argmax at every checkpoint slot, gold history.

    ``per_slot`` is over EVERY slot of every event and is directly comparable with the
    free-running ``checkpoint_acc``. ``trace`` is the final checkpoint's queried slot under the
    same forcing, and is the teacher-forced counterpart of the TRACE read. ``moving_slots`` is
    per_slot restricted to the slots whose value DIFFERS from the previous checkpoint's, which is
    the only part a copier does not already get for free.
    """
    import torch
    from sweep import _interleaved_slots

    examples = TK.generate(spec, "test", n=n, length=length)
    world, _r = TK.build_world(spec)
    k, m = spec.k, spec.n_objects_active
    agents, objs = list(world.agents[:k]), list(world.objects[:m])
    per = k + m
    prepped = []
    for ex in examples:
        inter = ex.meta.get("interleaved_prompt")
        if inter is None:
            return None
        toks, slots = _interleaved_slots(ex.prompt, inter)
        slotset = set(slots)
        ids, spos = [], []
        for j, w in enumerate(toks):
            e = tok.encode(w)
            if j in slotset:
                # a multi-token slot is not the single-token slot the free-running read
                # generates, so the two reads would not be scoring the same object
                if len(e) != 1:
                    return None
                spos.append(len(ids))
            ids += e
        prepped.append((ids, spos, [toks[s] for s in slots],
                        E.queried_slot_index(ex, k, m, agents, objs), ex.answer))
    hit = tot = 0
    mv_hit = mv_tot = 0
    tr_hit = tr_n = 0
    model.eval()
    with torch.no_grad():
        for b0 in range(0, len(prepped), batch):
            chunk = prepped[b0:b0 + batch]
            ml = max(len(c[0]) for c in chunk)
            inp = torch.full((len(chunk), ml), tok.pad_id, dtype=torch.long, device=device)
            for r, c in enumerate(chunk):
                inp[r, :len(c[0])] = torch.tensor(c[0], device=device)
            with torch.autocast(device, dtype=torch.bfloat16):
                logits = model(inp)
            pred = logits.float().argmax(-1).tolist()
            for r, (ids, spos, gold_ck, qi, ans) in enumerate(chunk):
                got = [tok.id_to_token.get(pred[r][p - 1], "<unk>") if p else None
                       for p in spos]
                for i, (g, a) in enumerate(zip(gold_ck, got)):
                    tot += 1
                    hit += int(g == a)
                    if i >= per and gold_ck[i - per] != g:
                        mv_tot += 1
                        mv_hit += int(g == a)
                if qi is None or len(got) < per:
                    continue
                tr_n += 1
                tr_hit += bool(TK.score_relaxed(Renderer.normalize(f"{got[-per + qi]}."),
                                                Renderer.normalize(ans)))
    model.train()
    return {"per_slot": hit / max(1, tot), "moving_slots": mv_hit / max(1, mv_tot),
            "n_moving": mv_tot, "trace": tr_hit / max(1, tr_n), "n": tr_n}


def main():
    ap = argparse.ArgumentParser(description="Teacher-forced checkpoint slots, per (arch, seed).")
    ap.add_argument("--ckpt_dir",
                    default="results/s5bind_v3_three_cell_depthmatched_20260801_ckpt")
    ap.add_argument("--out", default="results/s5bind_v3_teacher_forced_slots_20260801.json")
    ap.add_argument("--archs", default="gdp_hybrid,fprm,transformer")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--n", type=int, default=P.N_GUIDED)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()

    import torch
    specs = E.three_cell_specs(P.TRAIN_LENGTHS)
    world, renderer = TK.build_world(specs["composed"])
    tok = Tokenizer.build([world], renderer)
    out = {}
    if Path(a.out).exists():
        out = json.load(open(a.out)).get("rows", {})
    for arch in [x for x in ARCH_ORDER if x in [y.strip() for y in a.archs.split(",")]]:
        for seed in a.seeds:
            p = E.checkpoint_path(a.ckpt_dir, arch, seed)
            if not p.exists():
                print(f"  -- no checkpoint {p}; skipped", flush=True)
                continue
            model, blob = E.load_checkpoint(p, a.device)
            print(f"\n--- {arch} seed {seed} (loss "
                  f"{blob.get('provenance', {}).get('final_loss'):.4f}) ---", flush=True)
            for cell, L in CELLS:
                key = f"{arch}|{seed}|{cell}|{L}"
                if key in out:
                    continue
                t0 = time.time()
                b = a.batch
                while True:
                    try:
                        r = teacher_forced_slots(model, tok, specs[cell], L, a.n, a.device,
                                                 batch=b)
                        break
                    except torch.cuda.OutOfMemoryError:
                        torch.cuda.empty_cache()
                        if b <= 1:
                            raise
                        b //= 2
                if r is None:
                    print(f"     {cell}@{L}: unevaluable", flush=True)
                    continue
                out[key] = {**r, "arch": arch, "seed": seed, "cell": cell, "L": L,
                            "final_loss": blob.get("provenance", {}).get("final_loss")}
                print(f"     {cell:9s} L{L:<4d} per-slot={r['per_slot']:.3f} "
                      f"moving={r['moving_slots']:.3f} (n={r['n_moving']}) "
                      f"trace={r['trace']:.3f} [{time.time() - t0:.0f}s]", flush=True)
                Path(a.out).write_text(json.dumps(
                    {"written_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                     "ckpt_dir": a.ckpt_dir, "n": a.n, "rows": out}, indent=2, default=float))
            del model
            torch.cuda.empty_cache()
    print(f"\n=== done: {a.out} ===", flush=True)


if __name__ == "__main__":
    main()

"""THE READOUT STAGE — is the bounded-pad component answer a RECIPE problem or a real limit?

WHAT IS ALREADY MEASURED (``experiment_s5bind_v3_bounded_pad_20260802``). Under the bounded pad at
w = 2 with ``--pad_answer_docs`` the model's generated pad is BYTE-IDENTICAL to gold on every
component cell of every seed (``slot_acc`` 0.999-1.000 over 80- and 132-event streams), while the
ANSWER is at floor on 2 of 3 seeds. The state is written down correctly and cannot be read back.
That is a claim about a specific pair of numbers on a specific checkpoint, and it is testable
directly: take the floored checkpoint, hold the pad, and train NOTHING BUT the readout.

WHAT THIS MODULE ADDS, and it is one stage and no new spec:
    ``--finetune``   continue a saved checkpoint on documents whose loss is masked to the ANSWER,
                     at a ratio the flag sets, and re-read the same guided cells against the same
                     registered floors. If the answer comes up while ``slot_acc`` stays at 1.000,
                     the readout was a recipe problem and the arm is not blocked by the protocol.
                     If it does not, the pad is written but not addressable, which is a property of
                     the model's state and not of the training mix.
    ``--dump``       WHAT the floored model emits, as a histogram over its own answer strings with
                     the gold and the two nearest wrong references beside it. A single constant
                     token and a spread over the agent vocabulary are different failures and the
                     match number does not separate them.

THE LEVERS, in the order their cost puts them, all of them a change to the DOCUMENT MIX and none to
the format, the spec or the floor:
    ``--answer_ratio R``   R answer-masked copies per item against one full-loss pad copy. The
                           registered recipe is R = 1, where the answer carries 1 token against the
                           pad's 2L of a document that is itself one of two.
    ``--no_pad_docs``      drop the full-loss pad copy: a pure readout stage. The pad has no
                           supervision left, so ``slot_acc`` after the stage is the measurement
                           that says whether that mattered.
    ``--no_plain_docs``    drop the plain (unpadded) copy, which is the only document in the mix
                           that does not contain a pad at all.
    ``--answer_fresh``     draw the answer-masked copies from a DISJOINT item pool, so the stage
                           cannot be solved by memorising the item whose pad it just saw.

The stage runs on the SCHEDULE's last mix (state 0.15 / bind 0.15 / composed 0.70) so every cell is
under the same rule, which is what a three-cell comparison needs; ``--mix`` overrides it.

    finetune: .venv-train/bin/python scripts/experiment_s5bind_v3_readout_20260802.py --finetune \\
                  results/<run>_ckpt --seeds 1 2 --steps 3000 --answer_ratio 4
    dump:     ... --dump results/<run>_ckpt --seeds 1 --cells state@17,bind@31
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
from factworld.render import Renderer                                      # noqa: E402
from factworld.tokenizer import Tokenizer                                  # noqa: E402
import protocol_s5bind_v3_three_cell_20260731 as P                         # noqa: E402
import experiment_s5bind_v3_three_cell_local_20260731 as E                 # noqa: E402
import experiment_s5bind_v3_bounded_pad_20260802 as B                      # noqa: E402

FINAL_MIX = E.SCHEDULE[-1][2]
DEFAULT_CELLS = (("state", 17), ("state", 80), ("bind", 31), ("bind", 132), ("composed", 48))


def readout_documents(specs, weights, train_n, tok, fmt, *, answer_ratio=1, pad_docs=True,
                      plain_docs=True, fresh=False, group_masked=False):
    """``(encoded docs, prompt_lens)`` for a readout stage under a bounded pad.

    Same three document kinds ``B.stage_documents`` builds, with the answer-masked copy's COUNT and
    its ITEM SOURCE made into flags. ``fresh`` draws the answer-masked copies from items
    ``train_n`` .. ``2 * train_n`` of the same deterministic stream, which is disjoint from the pad
    copies' items and identically distributed.

    ``group_masked`` SORTS BY (is_masked, length) INSTEAD OF LENGTH, and it is the lever the
    document accounting picks out (``probe_s5bind_v3_answer_share_20260802``). ``train.run`` draws
    each batch as a contiguous slice of the sorted list and normalises the loss by the batch's total
    unmasked token count. An answer-masked pad document is BYTE-IDENTICAL to its full-loss twin, so
    a length sort puts the two adjacent and every batch mixes them; the answer's one token then
    competes with 2L of its twin's and takes 0.003 of the stage's loss, with NO batch in the stage
    reaching a 0.9 answer share. Grouping makes each batch either a tracking step or a readout step
    and takes the pad-prompt answer share to 0.333 at ratio 1 and 0.667 at ratio 4.
    """
    pairs = []
    for arm, share in sorted(weights.items()):
        n = int(round(train_n * share))
        if n <= 0:
            continue
        ags, obs = B.slot_order(specs[arm])
        pool = TK.generate(specs[arm], "train", n=(2 * n if fresh else n))
        base = pool[:n]
        alt = pool[n:] if fresh else base
        for e in base:
            if plain_docs:
                pairs.append((f"{e.prompt} {e.answer}", len(tok.encode(e.prompt))))
            doc = B.narrow_document(e, fmt, ags, obs)
            if doc is not None and pad_docs:
                pairs.append((doc, 1))
        for e in alt:
            doc = B.narrow_document(e, fmt, ags, obs)
            if doc is None:
                continue
            plen = len(tok.encode(doc[:-len(e.answer) - 1]))
            for _ in range(max(0, answer_ratio)):
                pairs.append((doc, plen))
    enc = [(tok.encode(t, add_eos=True)[:B.MAX_DOC_TOKENS], pl) for t, pl in pairs]
    enc.sort(key=(lambda x: (x[1] > 1, len(x[0]))) if group_masked else (lambda x: len(x[0])))
    return [a for a, _ in enc], [b for _, b in enc]


def read_cells(model, tok, specs, cells, guided_n, device, fmt, batch, floors):
    """Guided ``match``/``slot_acc`` per cell against the registered bounded-pad floor."""
    out = {}
    for cell, L in cells:
        a, sa, _t = B.bounded_free_run_batched(model, tok, specs[cell], L, guided_n, device, fmt,
                                               batch=batch)
        f = floors.get(f"{cell}@{L}")
        cl, z = P.clears(a, f, guided_n)
        out[f"{cell}@{L}"] = {"match": a, "slot_acc": sa, "floor": f, "clears": cl, "z": z}
        print(f"     {cell}@{L}: match={a:.3f} slot={sa:.3f} floor="
              + ("—" if f is None else f"{f:.4f}")
              + (f" z={z:.2f} " if z is not None else " ")
              + ("CLEARS" if cl else "at floor"), flush=True)
    return out


def floors_for(cells, guided_n, pad):
    """The bounded-pad floor at each cell, from ``validity`` and on the exact scored items."""
    from factworld import validity as V

    out = {}
    for cell, L in cells:
        spec = TK.CANONICAL[P.LOCAL_CELLS[cell]]
        k, m = spec.k, spec.n_objects_active
        pool = TK.generate(spec, "test", n=guided_n + P.N_SCORE, length=L)
        scored, big = pool[:guided_n], pool[guided_n:]
        named, query = V.s5_bind_v3_is_named(big), V.s5_bind_v3_query_kind(big)
        vals = []
        for items in (scored, big):
            ns, ng = V.s5_bind_v3_shape(items)
            vals.append(V.s5_bind_v3_pad_operative_floor(
                V.s5_bind_v3_pad_floors(items, k, m, named, query), k, m, ns, ng, named, query,
                pad=pad))
        out[f"{cell}@{L}"] = (None if all(v is None for v in vals)
                              else max(v for v in vals if v is not None))
        print(f"  floor (pad {pad}) {cell}@{L} = "
              + ("unfloorable" if out[f"{cell}@{L}"] is None else f"{out[f'{cell}@{L}']:.4f}"),
              flush=True)
    return out


def finetune(a):
    """Continue each saved checkpoint on ONE readout stage and re-read the guided cells."""
    import torch
    from factworld import train as T

    specs = E.three_cell_specs(P.TRAIN_LENGTHS)
    world, r = TK.build_world(specs["composed"])
    tok = Tokenizer.build([world], r)
    cells = [(c.split("@")[0], int(c.split("@")[1])) for c in a.cells.split(",")]
    pad = B.PAD_WIDTH[a.format]
    floors = floors_for(cells, a.guided_n, pad)
    mix = json.loads(a.mix) if a.mix else dict(FINAL_MIX)
    docs, plens = readout_documents(specs, mix, a.train_n, tok, a.format,
                                    answer_ratio=a.answer_ratio, pad_docs=not a.no_pad_docs,
                                    plain_docs=not a.no_plain_docs, fresh=a.answer_fresh,
                                    group_masked=a.group_masked)
    n_ans = sum(1 for p in plens if p > 1)
    print(f"  stage docs: {len(docs)} ({n_ans} answer-masked, {len(docs) - n_ans} full-loss), "
          f"sort={'grouped' if a.group_masked else 'length'}", flush=True)
    out = {"generated": datetime.now(timezone.utc).isoformat(), "pad_width": pad,
           "cfg": {**vars(a), "mix": mix, "n_docs": len(docs), "n_answer_docs": n_ans},
           "floors": floors, "runs": []}
    Path(a.out_prefix).parent.mkdir(parents=True, exist_ok=True)
    for arch in a.archs.split(","):
        for seed in a.seeds:
            pth = E.checkpoint_path(a.finetune, arch, seed)
            if not Path(pth).exists():
                print(f"  -- no checkpoint {pth}; skipped", flush=True)
                continue
            print(f"\n=== readout stage {arch} seed {seed} "
                  f"(ratio {a.answer_ratio}, pad_docs={not a.no_pad_docs}, "
                  f"plain_docs={not a.no_plain_docs}, fresh={a.answer_fresh}) ===", flush=True)
            model, blob = E.load_checkpoint(pth, a.device)
            before = read_cells(model, tok, specs, cells, a.guided_n, a.device, a.format,
                                a.guided_batch, floors)
            model.train()
            t0 = time.time()
            run = T.run(arch, tok, docs, [], steps=a.steps, batch=a.batch,
                        d_model=blob["build"]["d_model"], n_layers=blob["build"]["n_layers"],
                        n_heads=blob["build"]["n_heads"], d_ff=blob["build"]["d_ff"], lr=a.lr,
                        seed=seed, return_model=True, device=a.device, model=model,
                        use_short_conv=True, loss_log_interval=a.loss_log_interval,
                        prompt_lens=plens, warmup=a.warmup)
            model = run["model"]
            print(f"  -- {a.steps} steps, loss={run['final_loss']:.4f} "
                  f"[{time.time() - t0:.0f}s]", flush=True)
            after = read_cells(model, tok, specs, cells, a.guided_n, a.device, a.format,
                               a.guided_batch, floors)
            if a.save_ckpt:
                E.save_checkpoint(model, E.checkpoint_path(a.save_ckpt, arch, seed), arch=arch,
                                  seed=seed, stage="readout", build=blob["build"],
                                  provenance={"steps": a.steps, "n_docs": len(docs),
                                              "mix": mix, "lr": a.lr,
                                              "answer_ratio": a.answer_ratio,
                                              "group_masked": bool(a.group_masked),
                                              "from": str(pth),
                                              "final_loss": run["final_loss"]})
            del model
            torch.cuda.empty_cache()
            out["runs"].append({"arch": arch, "seed": seed, "before": before, "after": after,
                                "final_loss": run["final_loss"],
                                "loss_curve": [(int(s), float(v))
                                               for s, v in run.get("loss_curve", [])]})
            with open(f"{a.out_prefix}.json", "w") as f:
                json.dump(out, f, indent=1, default=float)
    print(f"\nwrote {a.out_prefix}.json")


def dump(a):
    """The floored model's own answer strings, as a histogram, with gold beside them."""
    import torch

    specs = E.three_cell_specs(P.TRAIN_LENGTHS)
    world, r = TK.build_world(specs["composed"])
    tok = Tokenizer.build([world], r)
    cells = [(c.split("@")[0], int(c.split("@")[1])) for c in a.cells.split(",")]
    for arch in a.archs.split(","):
        for seed in a.seeds:
            pth = E.checkpoint_path(a.dump, arch, seed)
            if not Path(pth).exists():
                print(f"  -- no checkpoint {pth}; skipped")
                continue
            model, _b = E.load_checkpoint(pth, a.device)
            for cell, L in cells:
                spec = specs[cell]
                ags, obs = B.slot_order(spec)
                examples = TK.generate(spec, "test", n=a.guided_n, length=L)
                prepped = []
                for ex in examples:
                    got = B.narrow_interleaved(ex, a.format, ags, obs)
                    toks, slots, gold = got
                    prepped.append((toks, slots, set(slots), gold, ex.answer))
                n_slots = len(prepped[0][1])
                preds, golds = [], []
                model.eval()
                with torch.no_grad():
                    for b0 in range(0, len(prepped), a.guided_batch):
                        chunk = prepped[b0:b0 + a.guided_batch]
                        ids = [[] for _ in chunk]
                        cursor = [0] * len(chunk)
                        for ordinal in range(n_slots + 1):
                            for i, (tk_, sl, ss, _g, _a) in enumerate(chunk):
                                limit = sl[ordinal] if ordinal < n_slots else len(tk_)
                                while cursor[i] < limit:
                                    if cursor[i] not in ss:
                                        ids[i] += tok.encode(tk_[cursor[i]])
                                    cursor[i] += 1
                            if ordinal < n_slots:
                                for i, tid in enumerate(E._batched_argmax(model, ids, tok,
                                                                          a.device)):
                                    ids[i].append(tid)
                                    cursor[i] += 1
                        outs = [[] for _ in chunk]
                        live = list(range(len(chunk)))
                        for _ in range(4):
                            if not live:
                                break
                            nxt = E._batched_argmax(model, [ids[i] for i in live], tok, a.device)
                            still = []
                            for i, tid in zip(live, nxt):
                                if tid == tok.eos_id:
                                    continue
                                ids[i].append(tid)
                                outs[i].append(tid)
                                still.append(i)
                            live = still
                        for i, (_t, _s, _ss, _g, ans) in enumerate(chunk):
                            preds.append(Renderer.normalize(tok.decode(outs[i])))
                            golds.append(Renderer.normalize(ans))
                cp, cg = Counter(preds), Counter(golds)
                hit = sum(1 for p, g in zip(preds, golds) if TK.score_relaxed(p, g))
                print(f"\n  {arch} s{seed} {cell}@{L}: match={hit / len(preds):.3f} "
                      f"n={len(preds)}  distinct_pred={len(cp)} distinct_gold={len(cg)}")
                print("    pred  " + "  ".join(f"{k!r}:{v}" for k, v in cp.most_common(8)))
                print("    gold  " + "  ".join(f"{k!r}:{v}" for k, v in cg.most_common(8)))
            del model
            torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--finetune", default=None, help="CKPT_DIR to continue")
    ap.add_argument("--dump", default=None, help="CKPT_DIR to decode and histogram")
    ap.add_argument("--save_ckpt", default=None)
    ap.add_argument("--archs", default="gdp_hybrid")
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2])
    ap.add_argument("--cells", default="state@17,state@80,bind@31,bind@132,composed@48")
    ap.add_argument("--format", default="moved2")
    ap.add_argument("--answer_ratio", type=int, default=4)
    ap.add_argument("--no_pad_docs", action="store_true")
    ap.add_argument("--no_plain_docs", action="store_true")
    ap.add_argument("--answer_fresh", action="store_true")
    ap.add_argument("--group_masked", action="store_true",
                    help="sort by (is_masked, length) so a batch is a tracking OR a readout step")
    ap.add_argument("--mix", default=None, help='JSON cell->share; default is the schedule\'s last')
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--train_n", type=int, default=20000)
    ap.add_argument("--guided_n", type=int, default=128)
    ap.add_argument("--guided_batch", type=int, default=128)
    ap.add_argument("--loss_log_interval", type=int, default=200)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out_prefix", default="results/s5bind_v3_readout_20260802")
    a = ap.parse_args()
    if a.dump:
        return dump(a)
    if a.finetune:
        return finetune(a)
    ap.error("one of --finetune or --dump")


if __name__ == "__main__":
    main()

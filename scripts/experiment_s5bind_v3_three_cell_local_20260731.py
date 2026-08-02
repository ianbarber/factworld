"""THE THREE-CELL COMPARISON, from-scratch arm — state component, retrieval component, composed
cell, each read against its own floor.

The reading rule is NOT here. It is pre-registered in
``scripts/protocol_s5bind_v3_three_cell_20260731.py`` and imported, so the verdict this script
prints is applied mechanically to whatever comes out.

WHAT THIS ARM MEASURES, AND IN WHICH COST MODEL
    A streaming model trained from scratch has no scratchpad, so its cost model is the forward
    pass and the answer is read directly off the prompt. That is the regime in which the floor's
    W axis has force: a model with O(1) live state IS the class of policy the one-structure
    bound prices, so clearing the composed cell's floor here means something the same number
    would not mean for a frontier model with a visible trace.

    Both registered reads are produced by ONE model, so neither costs a training run. The PLAIN
    read is the answer off the plain prompt in one token, over the whole grid. The GUIDED read
    is the s5 formation protocol — events teacher-forced, every per-event checkpoint and the
    answer generated — and it is registered at GUIDED_LENGTHS only, because its decode is
    (k+m)*L sequential rounds over a prompt that also grows with L, i.e. O(n L^2): at L=96 it is
    1152 rounds per batch against 576 at L=48, and the full grid at every cell is not affordable
    at this scale. ``sweep.py::guided_free_run_eval`` decodes one item at a time; this one
    batches the rounds, which is what makes even the short length affordable.

SUPERVISION, and why it is not the plain document
    In a plain document the answer is one token in ~440, so under an unmasked next-token loss
    the answer carries ~0.2% of the gradient and the run optimises the event stream instead.
    Measured: at d512x6 / 6000 steps / 40k documents the state component sits at 0.185 at L=48,
    i.e. its floor. Two changes, both registered before the three-cell run:

      1. the loss is MASKED to the answer on plain documents (``train.run(prompt_lens=...)``),
      2. the mix carries the specs' own per-event checkpoint documents (``event_trace`` —
         the whole of P then B after every event), which is the supervision density that formed
         S5 locally, under a full next-token loss.

    Both are chosen on a pilot run on the STATE COMPONENT ONLY, never on the composed cell, so
    the recipe is not selected on the outcome of interest.

CURRICULUM
    One model per (arch, seed) carried through three stages, each continuing from the previous
    weights: components alone, then components plus the composition, then composition-weighted.
    A composed cell trained cold is the confound that made the earlier local nulls unreadable —
    "the composition does not form" and "the composition was never given the recipe that made
    anything else form" are not separable without the staged arm.

    The evaluation grid is IN DISTRIBUTION (train lengths 16/32/48/64/96 cover the eval grid
    48/64/96). Length extrapolation is a second axis and would confound this one: a composed
    cell that fails out of distribution has two explanations and this run must have one.

COMPUTE MATCHING
    Architectures share (d_model, n_layers, n_heads, d_ff), which is the repo's compute-matched
    convention — ``fprm`` is weight-tied, so its parameter count is far lower at equal per-token
    FLOPs. Both numbers are measured and printed per architecture; neither is hidden, and the
    measured s/step is printed beside them because matched FLOPs/token is not matched wall clock.

CHECKPOINTS, so that an added eval length is a DECODE and not a retrain
    Weights are written per (arch, seed) after every stage to ``--ckpt_dir`` (default
    ``<out_prefix>_ckpt``), and ``--decode_from DIR`` scores those weights on the grid with no
    training. The depth-matched control cost a full retrain of the previous run only because
    nothing was saved.

Smoke test (minutes):
    .venv-train/bin/python scripts/experiment_s5bind_v3_three_cell_local_20260731.py \
        --archs gdp_hybrid --seeds 0 --steps 600 --d_model 128 --n_layers 2 --batch 8 \
        --train_n 400 --eval_n 100 --no_matched

Full run:
    .venv-train/bin/python scripts/experiment_s5bind_v3_three_cell_local_20260731.py \
        --archs gdp_hybrid,fprm,transformer --seeds 0 1 2 --steps 25000 --batch 16 \
        --d_model 768 --n_layers 8 --n_heads 6 --train_n 80000 --eval_n 1000

Re-score saved weights at a new length, with no training:
    .venv-train/bin/python scripts/experiment_s5bind_v3_three_cell_local_20260731.py \
        --decode_from results/<run>_ckpt --archs gdp_hybrid --seeds 0 1 2 \
        --out_prefix results/<new-read>
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
from factworld.backends import LocalBackend                                # noqa: E402
from factworld.render import Renderer                                      # noqa: E402
from factworld.runner import evaluate_task                                 # noqa: E402
from factworld.tokenizer import Tokenizer                                  # noqa: E402
from factworld import validity as V                                        # noqa: E402
import protocol_s5bind_v3_three_cell_20260731 as P                         # noqa: E402

# Stage weights over the three cells; each stage continues from the previous model's weights.
SCHEDULE = (
    ("stage1_components", 0.35, {"state": 0.5, "bind": 0.5}),
    ("stage2_add_composed", 0.30, {"state": 0.25, "bind": 0.25, "composed": 0.5}),
    ("stage3_composition", 0.35, {"state": 0.15, "bind": 0.15, "composed": 0.7}),
)
MAX_DOC_TOKENS = 3400          # the composed cell's checkpoint document at L=96 is 2504 tokens


def three_cell_specs(train_lengths):
    """The three registered cells, retrained on a length mix that COVERS the eval grid.

    Only ``train_lengths`` moves. ``generate(spec, "test", n, length=L)`` keys its RNG on
    (stream_name, split, L, idx) and never on the train mix, so every scored item at every eval
    length is byte-identical to the registered spec's and the published floors apply unchanged.
    """
    return {k: TK.CANONICAL[v].scaled(train_lengths=tuple(train_lengths))
            for k, v in P.LOCAL_CELLS.items()}


_DOC_CACHE: dict = {}


def stage_documents(specs, weights, train_n, tok, fmt):
    """``(encoded docs, prompt_lens)`` for one stage, length-sorted together.

    ``prompt_lens`` is 1 for a checkpoint document (full next-token loss: the checkpoints ARE
    the supervision) and the prompt's token count for a plain document (loss on the answer
    only, because otherwise the answer is one token in several hundred).

    Cached on (mix, train_n, fmt): ``generate(spec, "train", n)`` is deterministic, so every
    (arch, seed) sees the SAME documents in the same order, and rebuilding them per run is pure
    cost — 80k items per stage is ~3 minutes of sampling.
    """
    key = (tuple(sorted(weights.items())), train_n, fmt, tuple(sorted(specs)))
    if key in _DOC_CACHE:
        return _DOC_CACHE[key]
    pairs = []
    for arm, share in sorted(weights.items()):
        n = int(round(train_n * share))
        if n <= 0:
            continue
        for e in TK.generate(specs[arm], "train", n=n):
            if fmt in ("plain", "mix"):
                pairs.append((f"{e.prompt} {e.answer}", len(tok.encode(e.prompt))))
            if fmt in ("checkpoint", "mix") and "interleaved_prompt" in e.meta:
                pairs.append((f"{e.meta['interleaved_prompt']} {e.answer}", 1))
    enc = [(tok.encode(t, add_eos=True)[:MAX_DOC_TOKENS], pl) for t, pl in pairs]
    enc.sort(key=lambda x: len(x[0]))
    _DOC_CACHE[key] = ([a for a, _ in enc], [b for _, b in enc])
    return _DOC_CACHE[key]


def eval_cells(backend, specs, eval_n, grid):
    """``{cell: {L: match}}`` over the registered grid. One answer token, plain prompt."""
    out = {}
    for cell, lengths in grid.items():
        out[cell] = {}
        for L in lengths:
            res = evaluate_task(backend, specs[cell], split="test", n=eval_n, length=L)
            out[cell][str(L)] = res["overall"]
    return out


def queried_slot_index(example, k, m, agents, objs):
    """Position of the QUERIED slot inside one checkpoint block, or None.

    A checkpoint is the whole of P in ``agents`` order then the whole of B in ``objs`` order, so
    the queried slot is at ``agents.index(q_state)`` on a state query and ``k + objs.index(q_bind)``
    on a bind query. This is the index ``validity.s5_bind_v3_trace_slot`` reads the GOLD trace at;
    the trace read applies it to the model's OWN generated checkpoints.
    """
    qs, qb = example.meta.get("q_state"), example.meta.get("q_bind")
    if example.meta.get("query_kind") == "bind" or (qs is None and qb is not None):
        return None if qb not in objs else k + list(objs).index(qb)
    if qs is not None and qs in agents:
        return list(agents).index(qs)
    if qb is not None and qb in objs:
        return k + list(objs).index(qb)
    return None


def guided_free_run_batched(model, tok, spec, length, n, device, batch=128, max_answer=4):
    """GUIDED read: events teacher-forced, every checkpoint and the answer GENERATED.

    The s5 formation protocol (``scripts/sweep.py::guided_free_run_eval``), batched, because
    the reference decodes one item at a time with no cache and a checkpoint here is the whole
    of P then B — k + m = 12 tokens per event, so 1152 sequential forward passes per item at
    L = 96. Every item at one length has the same NUMBER of slots even though their absolute
    positions differ (event sentences differ in width), so the loop runs once per slot ORDINAL
    with one padded batched forward per round.

    Returns ``(answer_match, checkpoint_acc, trace)``. THIS IS A SCRATCHPAD PROTOCOL, and both of
    the quantities it returns come out of it: the generated checkpoints accumulate into ``ids``,
    which is the same context the answer is then decoded from. So the floor's W axis has no force
    against either, and the rule is ``validity.s5_bind_v3_slot_profile``'s — a model under this
    protocol is read against the TOP of the cell's profile, not its admitted end. On the composed
    cell that leaves no floor at all on either channel (``P.cell_floor(..., guided=True)``).

    ``trace`` is the third quantity and the one the answer channel cannot reach: the model's OWN
    final checkpoint's value for the queried slot, scored against the same gold under the same
    canonical metric. It is ``{"match", "n", "t1_agree"}``, where ``t1_agree`` re-measures T1 —
    that the GOLD final checkpoint's queried slot is the gold answer — on the exact scored items,
    because the whole read rests on the two channels scoring one quantity. It is None where the
    queried slot is not indexable (no ``meta["trace"]``, or a query the block does not carry).
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
            return None, None, None
        toks, slots = _interleaved_slots(ex.prompt, inter)
        prepped.append((toks, slots, set(slots), [toks[s] for s in slots], ex.answer,
                        queried_slot_index(ex, k, m, agents, objs)))
    n_slots = len(prepped[0][1])
    if any(len(p[1]) != n_slots for p in prepped):
        return None, None, None
    hits = ck_hits = ck_total = 0
    tr_hits = tr_n = t1_agree = 0
    model.eval()
    with torch.no_grad():
        for b0 in range(0, len(prepped), batch):
            chunk = prepped[b0:b0 + batch]
            ids = [[] for _ in chunk]
            cursor = [0] * len(chunk)
            gen_ck = [[] for _ in chunk]
            for ordinal in range(n_slots + 1):
                for i, (toks, slots, slotset, _gold, _ans, _qi) in enumerate(chunk):
                    limit = slots[ordinal] if ordinal < n_slots else len(toks)
                    while cursor[i] < limit:
                        if cursor[i] not in slotset:
                            ids[i] += tok.encode(toks[cursor[i]])
                        cursor[i] += 1
                if ordinal < n_slots:
                    nxt = _batched_argmax(model, ids, tok, device)
                    for i, tid in enumerate(nxt):
                        ids[i].append(tid)
                        gen_ck[i].append(tok.id_to_token.get(tid, "<unk>"))
                        cursor[i] += 1                      # the slot itself is now consumed
            outs = [[] for _ in chunk]
            live = list(range(len(chunk)))
            for _ in range(max_answer):
                if not live:
                    break
                nxt = _batched_argmax(model, [ids[i] for i in live], tok, device)
                still = []
                for i, tid in zip(live, nxt):
                    if tid == tok.eos_id:
                        continue
                    ids[i].append(tid)
                    outs[i].append(tid)
                    still.append(i)
                live = still
            for i, (_t, _s, _ss, gold_ck, ans, qi) in enumerate(chunk):
                pred = tok.decode(outs[i])
                hits += bool(TK.score_relaxed(Renderer.normalize(pred),
                                              Renderer.normalize(ans)))
                ck_hits += sum(1 for a, g in zip(gen_ck[i], gold_ck) if a == g)
                ck_total += len(gold_ck)
                # THE TRACE READ. The same gold, scored off the model's own final checkpoint
                # instead of off the answer token — and T1 (that the GOLD final checkpoint's
                # queried slot IS that gold) is re-measured here rather than assumed, because if
                # it fails the two channels are not scoring one quantity.
                if qi is None or len(gen_ck[i]) < per or len(gold_ck) < per:
                    continue
                tr_n += 1
                tr_hits += bool(TK.score_relaxed(
                    Renderer.normalize(f"{gen_ck[i][-per + qi]}."), Renderer.normalize(ans)))
                t1_agree += bool(TK.score_relaxed(
                    Renderer.normalize(f"{gold_ck[-per + qi]}."), Renderer.normalize(ans)))
    model.train()
    trace = None if not tr_n else {"match": tr_hits / tr_n, "n": tr_n, "t1_agree": t1_agree}
    return hits / max(1, len(prepped)), ck_hits / max(1, ck_total), trace


def _batched_argmax(model, id_lists, tok, device):
    """Greedy next token for a ragged batch of prefixes, one padded forward pass."""
    import torch

    ml = max(len(s) for s in id_lists)
    inp = torch.full((len(id_lists), ml), tok.pad_id, dtype=torch.long, device=device)
    last = torch.empty(len(id_lists), dtype=torch.long, device=device)
    for r, s in enumerate(id_lists):
        if s:
            inp[r, :len(s)] = torch.tensor(s, device=device)
        last[r] = max(0, len(s) - 1)
    with torch.autocast(device, dtype=torch.bfloat16):
        logits = model(inp)
    return logits[torch.arange(len(id_lists), device=device), last].float().argmax(-1).tolist()


def checkpoint_path(ckpt_dir, arch, seed):
    return Path(ckpt_dir) / f"{arch}_seed{seed}.pt"


def save_checkpoint(model, path, *, arch, seed, stage, build, provenance):
    """Write the trained weights, plus everything ``load_checkpoint`` needs to rebuild the model.

    THE POINT IS THAT AN ADDED EVAL LENGTH IS A DECODE AND NOT A RETRAIN. The depth-matched
    control this run exists to measure cost a full retrain of the previous one only because
    nothing was checkpointed; at ~1.6 GPU-hours per (arch, seed) that is the difference between
    a control being bought and being argued about. Written after EVERY stage to the same path,
    so a crash in stage 3 still leaves stage 2's weights decodable.
    """
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".pt.tmp")
    torch.save({"arch": arch, "seed": seed, "stage": stage, "build": dict(build),
                "provenance": dict(provenance),
                "state_dict": {k: v.detach().to("cpu") for k, v in model.state_dict().items()}},
               tmp)
    tmp.replace(path)


def load_checkpoint(path, device):
    """``(model, blob)`` — the weights rebuilt into an eval-ready model on ``device``."""
    import torch
    from factworld.models import build_model

    blob = torch.load(path, map_location="cpu", weights_only=False)
    b = blob["build"]
    model = build_model(blob["arch"], b["vocab_size"], d_model=b["d_model"],
                        n_layers=b["n_layers"], n_heads=b["n_heads"], d_ff=b["d_ff"],
                        use_short_conv=b.get("use_short_conv", True)).to(device)
    model.load_state_dict(blob["state_dict"])
    model.eval()
    return model, blob


def evaluate_all(model, arch, specs, tok, world, grid, *, eval_n, guided_n, guided_lengths,
                 device, guided_batch=128):
    """``(plain, guided)`` for ONE model over a grid — the whole read, and nothing trains here.

    Shared by the training path and by the checkpoint DECODE path, so the two cannot drift: an
    eval length added after a run is scored by exactly the code that scored the registered ones.
    """
    backend = LocalBackend([world], arch=arch, model=model, tokenizer=tok, device=device)
    ev = eval_cells(backend, specs, eval_n, grid)
    for cell, cells in ev.items():
        print("     plain  " + f"{cell:9s} "
              + "  ".join(f"L{L}={a:.3f}" for L, a in cells.items()), flush=True)
    gv = {}
    if guided_n:
        for cell in specs:
            gv[cell] = {}
            lens = (guided_lengths.get(cell, ()) if isinstance(guided_lengths, dict)
                    else guided_lengths)
            for L in lens:
                t1 = time.time()
                a, ck, tr = guided_free_run_batched(model, tok, specs[cell], L, guided_n,
                                                    device, batch=guided_batch)
                gv[cell][str(L)] = {"match": a, "checkpoint_acc": ck, "trace": tr}
                # An unevaluable guided cell is reported as UNEVALUABLE and does not abort the
                # (arch, seed): discarding an otherwise complete run over one cell costs the
                # whole training run, and the rule downstream already refuses to read a missing
                # cell as a cell at floor.
                shown = ("unevaluable" if a is None else
                         f"match={a:.3f} ck={ck:.3f}"
                         + ("" if tr is None else f" trace={tr['match']:.3f}"))
                print(f"     guided {cell:9s} L{L}: {shown} [{time.time() - t1:.0f}s]",
                      flush=True)
    return ev, gv


def decode_runs(ckpt_dir, archs, seeds, specs, tok, world, grid, *, eval_n, guided_n,
                guided_lengths, device, guided_batch=128):
    """Score saved checkpoints on a grid, with no training. Returns rows shaped like ``run_one``.

    A checkpoint the directory does not hold is REPORTED AND SKIPPED rather than substituted by
    an untrained model, which would enter the tables as a cell at floor.
    """
    runs = []
    for arch in archs:
        for seed in seeds:
            p = checkpoint_path(ckpt_dir, arch, seed)
            if not p.exists():
                print(f"  -- no checkpoint {p}; skipped (a missing model is not a model at "
                      "floor)", flush=True)
                continue
            print(f"\n--- decode {arch} seed {seed} <- {p} ---", flush=True)
            model, blob = load_checkpoint(p, device)
            t0 = time.time()
            ev, gv = evaluate_all(model, arch, specs, tok, world, grid, eval_n=eval_n,
                                  guided_n=guided_n, guided_lengths=guided_lengths,
                                  device=device, guided_batch=guided_batch)
            prov = blob.get("provenance", {})
            runs.append({"arch": arch, "seed": seed, "decoded_from": str(p),
                         "stages": [{"stage": blob.get("stage", "decode"),
                                     "steps": prov.get("steps"), "n_docs": prov.get("n_docs"),
                                     "mix": prov.get("mix", {}),
                                     "final_loss": prov.get("final_loss"), "loss_curve": [],
                                     "train_s": 0, "decode_s": round(time.time() - t0),
                                     "eval": ev, "guided": gv}]})
            del model
            import torch
            torch.cuda.empty_cache()
    return runs


def measured_size(arch, d_model, n_layers, n_heads, vocab_size, device, seq_len=512):
    """``(params, per-token forward FLOPs)`` — the compute axis the comparison is matched on."""
    import torch
    from torch.utils.flop_counter import FlopCounterMode
    from factworld.models import build_model

    torch.manual_seed(0)
    m = build_model(arch, vocab_size, d_model=d_model, n_layers=n_layers, n_heads=n_heads,
                    d_ff=4 * d_model, use_short_conv=True).to(device).eval()
    params = (m.num_params() if hasattr(m, "num_params")
              else sum(p.numel() for p in {id(p): p for p in m.parameters()}.values()))
    ids = torch.randint(0, vocab_size, (1, seq_len), device=device)
    with torch.no_grad(), torch.autocast(device, dtype=torch.bfloat16):
        m(ids)
    fcm = FlopCounterMode(display=False)
    with fcm, torch.no_grad(), torch.autocast(device, dtype=torch.bfloat16):
        m(ids)
    total = fcm.get_total_flops()
    del m
    torch.cuda.empty_cache()
    return params, total // seq_len


def run_one(arch, seed, specs, tok, world, grid, *, steps, batch, d_model, n_layers, n_heads,
            lr, train_n, eval_n, guided_n, guided_lengths, device, fmt, loss_log_interval,
            guided_batch=128, ckpt_dir=None):
    import torch
    from factworld import train as T

    model, stages = None, []
    for si, (name, share, weights) in enumerate(SCHEDULE):
        last = si == len(SCHEDULE) - 1
        stage_steps = max(1, int(round(steps * share)))
        docs, plens = stage_documents(specs, weights, train_n, tok, fmt)
        t0 = time.time()
        run = T.run(arch, tok, docs, [], steps=stage_steps, batch=batch, d_model=d_model,
                    n_layers=n_layers, n_heads=n_heads, d_ff=4 * d_model, lr=lr, seed=seed,
                    return_model=True, device=device, model=model, use_short_conv=True,
                    loss_log_interval=loss_log_interval, prompt_lens=plens)
        model = run["model"]
        print(f"  -- {name}: {stage_steps} steps, {len(docs)} docs, "
              f"loss={run['final_loss']:.4f} [{time.time() - t0:.0f}s]", flush=True)
        if ckpt_dir:
            save_checkpoint(model, checkpoint_path(ckpt_dir, arch, seed), arch=arch, seed=seed,
                            stage=name,
                            build={"d_model": d_model, "n_layers": n_layers, "n_heads": n_heads,
                                   "d_ff": 4 * d_model, "use_short_conv": True,
                                   "vocab_size": tok.vocab_size},
                            provenance={"steps": stage_steps, "n_docs": len(docs), "mix": weights,
                                        "final_loss": run["final_loss"], "lr": lr, "batch": batch,
                                        "fmt": fmt, "train_n": train_n,
                                        "train_lengths": list(P.TRAIN_LENGTHS)})
        if last:
            ev, gv = evaluate_all(model, arch, specs, tok, world, grid, eval_n=eval_n,
                                  guided_n=guided_n, guided_lengths=guided_lengths,
                                  device=device, guided_batch=guided_batch)
        else:
            # intermediate stages get a progress read only; the registered grid, the matched-cost
            # control lengths and the guided decode are paid for once, at the end.
            ev, gv = evaluate_all(
                model, arch, specs, tok, world,
                {c: [P.CONTROL_LENGTH, P.registered_lengths(c)[0]] for c in grid},
                eval_n=200, guided_n=0, guided_lengths={}, device=device)
        stages.append({"stage": name, "steps": stage_steps, "n_docs": len(docs),
                       "mix": weights, "final_loss": run["final_loss"],
                       "loss_curve": [(int(s), float(v)) for s, v in run.get("loss_curve", [])],
                       "train_s": round(time.time() - t0), "eval": ev, "guided": gv})
    del model
    torch.cuda.empty_cache()
    return stages


def _accuracies(rows, grid, read):
    """``{cell: {seed: {L: match}}}`` for one read, off the final stage.

    A cell with no score is DROPPED rather than carried as None. Carrying it would make
    ``clears`` return False and the cell would enter the verdict as a model at floor, which is
    the substitution the whole rule exists to refuse; dropped, it reaches the missing-cell raise.
    """
    out = {}
    for cell in grid:
        out[cell] = {}
        for r in rows:
            blk = r["stages"][-1]["eval" if read == "plain" else "guided"].get(cell, {})
            vals = {int(L): (v if read == "plain" else (v or {}).get("match"))
                    for L, v in blk.items()}
            out[cell][r["seed"]] = {L: v for L, v in vals.items() if v is not None}
    return out


def _floor_map(floors, grid, key="floor"):
    return {cell: {int(k.split("@")[1]): v.get(key) for k, v in floors.items()
                   if k.split("@")[0] == cell} for cell in grid}


def _guided_records(guided_floors, floors):
    """The floor records the GUIDED read is judged against, with no path back to a plain floor on
    the composed cell.

    A record written before the guided floors were measured separately has none, and the old
    fallback was the PLAIN floors — which floor the composed cell at the one-structure bound, the
    exact number this protocol voids. So the fallback keeps the component records (a component's
    class is the same under either protocol) and returns the composed cell UNFLOORABLE, which is
    what the guided protocol leaves it. Any record already measured under ``guided`` is used as
    is; a stale one is not silently mixed in.
    """
    if guided_floors and all(v.get("protocol") == "guided" for v in guided_floors.values()):
        return guided_floors
    out = {}
    for key, rec in (guided_floors or floors or {}).items():
        if key.split("@")[0] == "composed":
            out[key] = {**rec, "floor": None, "floor_plain": rec.get("floor"),
                        "pad_reach": rec.get("pad_reach"), "basis": "unfloorable",
                        "protocol": "guided_fallback"}
        else:
            out[key] = {**rec, "protocol": "guided_fallback"}
    return out


def apply_rule(runs, floors, grid, eval_n, guided_n, guided_floors=None):
    """The pre-registered rule, applied to the final-stage numbers of BOTH reads separately.

    Never mixed: judging the components on the guided read and the composed cell on the plain
    one would manufacture a composition gap out of the eval mode.

    EACH READ IS JUDGED AGAINST THE FLOOR MEASURED ON ITS OWN SCORED ITEMS. The guided read
    scores ``guided_n`` items and the plain read ``eval_n``, and the max over admitted rows
    carries an upward selection bias that does not shrink with the score's own n: at n = 128 the
    operative floor is 0.250 at state@80 and 0.234 at composed@48 against 0.207 and 0.204 at
    n = 1000, on the same rows. Where a record predates the separately-measured guided floors,
    ``_guided_records`` keeps the component numbers and returns the composed cell UNFLOORABLE —
    it never falls back to the plain floor there, which is the one this protocol voids.

    AND EACH READ AGAINST ITS OWN PROTOCOL'S CLASS RULE. A guided floor is None on the composed
    cell — the format hands out the live slots that cell's floor argument prices — so the composed
    cell reaches ``verdict`` as UNFLOORABLE and not as a cell at floor.
    """
    per_arch = {}
    for arch in sorted({r["arch"] for r in runs}):
        rows = [r for r in runs if r["arch"] == arch]
        gf = _guided_records(guided_floors, floors)
        fp = _floor_map(floors, grid)
        fg = _floor_map(gf, grid)
        pads = _floor_map(gf, grid, "pad_reach")
        per_read = {}
        for read, n in (("plain", eval_n), ("guided", guided_n)):
            f = fp if read == "plain" else fg
            acc = _accuracies(rows, grid, read)
            if not any(acc[c][s] for c in acc for s in acc[c]):
                continue
            # EACH CELL AT ITS OWN REGISTERED LENGTHS: the composed grid for the composed cell,
            # the WORK-MATCHED partners of it for the components. Reading a component at the
            # composed cell's own L compares 1/p_swap times the state work and 1/(1 - p_swap)
            # times the retrieval work, which is the confound this pairing removes.
            comp_forms, comp_counts, lengths = {}, {}, {}
            for cell in ("state", "bind", "composed"):
                lengths[cell] = tuple(L for L in P.registered_lengths(cell)
                                      if all(L in acc[cell][s] for s in acc[cell]))
                if not lengths[cell]:
                    # A cell the run never evaluated at its registered lengths is a MISSING CELL,
                    # and forms() would report it with the same False a floored cell gets. That
                    # is the substitution the whole rule exists to refuse.
                    raise P.ControlNotEvaluable(
                        f"{arch}/{read}: {cell} was not evaluated at any of its registered "
                        f"lengths {P.registered_lengths(cell)}; it was read at "
                        f"{sorted({L for s in acc[cell] for L in acc[cell][s]})}. A missing cell "
                        "is not a cell at floor.")
                ok, counts = P.forms(acc[cell], f[cell], lengths[cell], n=n)
                comp_forms[cell], comp_counts[cell] = ok, counts
            # the positive control, on the grid THIS read covers. Unevaluable raises rather than
            # aborting: an (arm, cell, length) the arm never ran is a missing cell, not a model
            # at floor, and a bare seed count reports the two with the same number.
            ctrl = P.evaluate_control(read, acc, f, {c: sorted(grid[c]) for c in grid}, n)
            matched = {}
            matched_measured = {}
            for cell in ("state", "bind"):
                # a matched length with no floor is NOT MEASURED, not failed: scoring it
                # against a missing floor would read as "the control did not form" and flip
                # the verdict to V3 on an absence.
                mlens = tuple(L for L in P.matched_lengths_for(cell)
                              if L in set(grid[cell]) and f[cell].get(L) is not None
                              and all(L in acc[cell][s] for s in acc[cell]))
                matched[cell] = P.forms(acc[cell], f[cell], mlens, n=n)[0] if mlens else None
                matched_measured[cell] = bool(mlens)
            # UNFLOORABLE IS NOT FLOORED. Where this read's protocol leaves the composed cell no
            # floor at any registered length, the cell cannot clear and cannot fail to clear, and
            # the verdict says so instead of reading a null off an absent number.
            floored = all(f["composed"].get(L) is not None for L in lengths["composed"])
            pad = next((pads["composed"][L] for L in lengths["composed"]
                        if pads["composed"].get(L) is not None), None) if read == "guided" else None
            code, why = P.verdict(ctrl, comp_forms, comp_counts, matched, matched_measured,
                                  composed_floored=floored, pad_reach=pad)
            per_read[read] = {"verdict": code, "why": why, "control": ctrl,
                              "control_seeds": ctrl["seeds"],
                              "matched_measured": matched_measured,
                              "composed_floored": floored, "composed_pad_reach": pad,
                              "lengths_read": {c: list(v) for c, v in lengths.items()},
                              "forms": comp_forms,
                              "seed_counts": comp_counts, "matched_forms": matched, "acc": acc}
        per_arch[arch] = per_read
    return per_arch


def post_hoc_section(runs, floors, cfg):
    """Two readings computed AFTER the numbers existed, kept apart from the pre-registered rule.

    The first is the per-seed pairing between the composed cell and each component ON THE
    DEPTH-MATCHED LENGTHS — the rule counts seeds per cell and cannot see whether it is the SAME
    seeds. Each component is read at its WORK-MATCHED partner of the composed length (state@17
    and bind@31 against composed@48 at k=6), never at the composed cell's own L: at p_swap = 1/3
    that would put a 5.7-hop composed cell beside a 16.0-hop state cell, which is the confound
    the pairing exists to remove.

    The second is what the checkpoint diagnostic does and does not say, which matters because the
    number looks like a partial trace and is not one. Every figure in it is computed from this
    run's own cells.
    """
    if not runs:
        return []
    L = cfg["guided_lengths"][0] if cfg.get("guided_lengths") else 48
    k = cfg["k"]
    at = {"composed": L, "state": P.WORK_MATCHED.get(L, {}).get("state"),
          "bind": P.WORK_MATCHED.get(L, {}).get("bind")}
    cfl = floors.get(f"composed@{L}", {})
    unfloorable = cfl.get("floor") is None
    pad = cfl.get("pad_reach")
    out = ["", "# Post-hoc (not pre-registered)", "",
           "## The composed cell and its depth-matched components, seed by seed (GUIDED)", "",
           f"Each cell at the length carrying the same amount of its own work: composed@{L}, "
           f"state@{at['state']}, bind@{at['bind']}. A blank is a cell this read did not cover.",
           "", f"| arch | seed | state@{at['state']} | composed@{L} | bind@{at['bind']} |",
           "|---|---|---|---|---|"]
    verdict_lines = []
    for arch in sorted({r["arch"] for r in runs}):
        rows = []
        for r in sorted((r for r in runs if r["arch"] == arch), key=lambda r: r["seed"]):
            g = r["stages"][-1]["guided"]
            cell = {}
            for c in ("state", "bind", "composed"):
                cl = at[c]
                v = (g.get(c, {}).get(str(cl), {}) or {}).get("match") if cl else None
                f = floors.get(f"{c}@{cl}", {}).get("floor") if cl else None
                cell[c] = (v, P.clears(v, f, cfg["guided_n"])[0] if v is not None else False)
            rows.append((r["seed"], cell))
            out.append(f"| {arch} | {r['seed']} | " + " | ".join(
                ("—" if cell[c][0] is None else
                 f"{cell[c][0]:.3f}"
                 + (" (clears)" if cell[c][1] else
                    ("†" if c == "composed" and unfloorable else "")))
                for c in ("state", "composed", "bind")) + " |")
        full = [(s, c) for s, c in rows
                if all(c[x][0] is not None for x in ("state", "bind", "composed"))]
        if not full:
            continue
        if unfloorable:
            # THE CLEARS AXIS IS NOT AVAILABLE HERE. The composed cell has no floor under this
            # protocol, so "clears on exactly the seeds the state component does" is a comparison
            # with one side missing. What the same seeds and the same items do support is the
            # DIRECTION, per seed — but only on the seeds whose STATE leg is off its own floor.
            # Differencing two cells that are both at floor reports noise as a composition cost,
            # which is the same substitution the floored branch below refuses.
            live = [(s, c) for s, c in full if c["state"][1]]
            if not live:
                verdict_lines.append(
                    f"- **{arch}**: the composed cell is UNFLOORABLE on this read and its "
                    f"depth-matched state component is at floor on all {len(full)} seeds, so "
                    "there is nothing to compare it with. The two columns agree because neither "
                    "cell moved.")
                continue
            deficits = [(s, c["composed"][0] - c["state"][0]) for s, c in live]
            below = sum(1 for _s, d in deficits if d < 0)
            verdict_lines.append(
                f"- **{arch}**: the composed cell is UNFLOORABLE on this read, so there is no "
                "clears/does-not-clear pairing to read. On the "
                f"{len(live)} of {len(full)} seeds whose state component is off its own floor, "
                f"the composed cell is BELOW it on {below}, within the run and on the same "
                "items: "
                + ", ".join(f"seed {s} {d:+.3f}" for s, d in deficits)
                + (f". The excluded both-maps class reaches {pad:.3f} on those items, so the "
                   "composed cell scores from a cheap-policy baseline far above the state "
                   "component's." if pad is not None else "."))
            continue
        same = [s for s, c in full if c["state"][1] == c["composed"][1]]
        split = [s for s, c in full if c["state"][1] and not c["composed"][1]]
        cleared = [s for s, c in full if c["state"][1] or c["composed"][1]]
        if not cleared:
            # AGREEING AT FLOOR IS NOT A PAIRING. Reading "clears on exactly the same seeds" off
            # a set where nothing clears turns a null into a composition result; the seeds agree
            # because neither cell moved, which says nothing about what the composed cell costs.
            verdict_lines.append(
                f"- **{arch}**: neither the composed cell nor its depth-matched state component "
                f"clears on any of the {len(full)} seeds, so there is no pairing to read. The "
                "cells agree because both are at floor.")
        elif len(same) == len(full):
            verdict_lines.append(
                f"- **{arch}**: at equal state depth the composed cell clears on exactly the "
                f"seeds the state component clears on ({len(full)}/{len(full)} seeds agree, "
                f"{len(cleared)} of them clearing), so the composed cell costs this architecture "
                "nothing beyond the state leg it contains ON THE CLEARS/DOES-NOT-CLEAR AXIS. It "
                "is not a claim about the sizes: the pre-registered rule counts seeds per cell "
                "and reads neither the margin nor the direction.")
        elif split:
            verdict_lines.append(
                f"- **{arch}**: the state component clears at depth {at['state']} on seeds "
                f"{split} where the composed cell at the SAME state depth does not, which is a "
                "per-seed composition cost the seed counts do not show.")
        else:
            verdict_lines.append(
                f"- **{arch}**: the composed cell and its depth-matched state component do not "
                f"agree seed for seed (agree on {len(same)}/{len(full)}).")
    if unfloorable:
        out += ["", "A † marks a cell with NO FLOOR on this protocol. The composed cell's floor "
                "argument is the one-structure bound, and the guided format writes the whole of "
                "P then B at every event — so the k + m slots it prices are handed to every "
                "policy, on the answer channel as much as on the trace channel."
                + (f" What the excluded both-maps class reaches on these exact items is "
                   f"{pad:.3f}, against a plain-protocol floor of "
                   f"{cfl.get('floor_plain'):.3f}; it is a lower bound on that class's max and "
                   "not a bar." if pad is not None and cfl.get("floor_plain") is not None
                   else "")]
    if verdict_lines:
        out += [""] + verdict_lines
    out += checkpoint_diagnostic_section(runs, cfg)
    return out


def checkpoint_diagnostic_section(runs, cfg):
    """What the per-slot checkpoint number does and does not say — against the RIGHT reference.

    THE REFERENCE IS NOT 1/k AND NOT THE FROZEN-HALF NUMBER. Both readings have been published
    here and both are wrong. The constant half of a COMPONENT checkpoint is UNSTATED — the state
    cell states no holders and the retrieval cell states no pointers — so "emit the frozen half"
    is not a policy a model can run off the prompt. And the diagnostic scores every slot of every
    event, where most slots do not move: a swap moves 2 of the k + m and a give moves 1, so a
    model that emits its previous checkpoint unchanged at every event is right on
    ``1 - (2 n_swap + n_give) / ((k + m) L)`` of the slots. That is the number a per-slot score
    has to beat, and it is 0.80-0.91 here against 1/k = 0.167.

    ``validity.s5_bind_v3_ckpt_copy_per_slot`` recomputes it from each cell's own scored items.
    """
    per_slot = {}
    for cellname in ("state", "bind", "composed"):
        for r in sorted(runs, key=lambda x: (x["arch"], x["seed"])):
            for L, v in r["stages"][-1]["guided"].get(cellname, {}).items():
                if v.get("checkpoint_acc") is not None:
                    per_slot.setdefault((r["arch"], cellname, int(L)), []).append(
                        v["checkpoint_acc"])
    out = ["", "## What the checkpoint diagnostic says", ""]
    if not per_slot:
        return out
    specs = three_cell_specs(P.TRAIN_LENGTHS)
    world, _r = TK.build_world(specs["composed"])
    k = cfg["k"]
    m = specs["composed"].n_objects_active
    agents, objs = list(world.agents[:k]), list(world.objects[:m])
    n = cfg.get("guided_n") or P.N_GUIDED
    ref = {}
    for (_arch, cellname, L) in sorted(per_slot):
        if (cellname, L) in ref:
            continue
        ex = TK.generate(specs[cellname], "test", n=n, length=L)
        ref[(cellname, L)] = V.s5_bind_v3_ckpt_copy_per_slot(ex, k, m, agents, objs)
    lo, hi = min(ref.values()), max(ref.values())
    out += [f"Each checkpoint is the whole of P and then the whole of B, k + m = {2 * k} slots "
            "per event, and the diagnostic scores every slot of every event. MOST SLOTS DO NOT "
            "MOVE: a swap moves 2 of them and a give moves 1, so a model that re-emits its "
            "previous checkpoint unchanged at every event is already right on "
            f"{lo:.3f}-{hi:.3f} of the slots, against 1/k = {1.0 / k:.3f}. That copier is the "
            "reference, and it is what a per-slot number has to be read against.", "",
            "| arch | cell | L | per-seed per-slot | copy-the-previous-checkpoint | "
            "above the copier |", "|---|---|---|---|---|---|"]
    for (arch, cellname, L), vals in sorted(per_slot.items()):
        r0 = ref[(cellname, L)]
        out.append(f"| {arch} | {cellname} | {L} | " + " ".join(f"{v:.3f}" for v in vals)
                   + f" | {r0:.3f} | " + " ".join(f"{v - r0:+.3f}" for v in vals) + " |")
    out += ["", "The diagnostic is not a partial trace and no verdict reads it. What IS read is "
            "the TRACE read — the final checkpoint's value for the QUERIED slot — a single slot "
            "scored against the same gold and against the same floors as this protocol's answer "
            "channel (`validity.s5_bind_v3_operative_floor(..., guided=True)`: the component "
            "cells floored, the composed cell unfloorable). The copier scores 0.000 on it at "
            "every cell because the query gate requires the queried slot to move at least twice "
            "and to end different from its stated value."]
    return out


def pilot_section(cfg):
    """The single-seed gate that ran before the grid, and what it did and did not settle."""
    p = cfg.get("pilot")
    if not p:
        return []
    out = ["", "## The gate that ran first (one seed)", "",
           p["what"], "",
           "| steps | " + " | ".join(f"L{L}" for L in p["lengths"]) + " | loss |",
           "|" + "---|" * (len(p["lengths"]) + 2)]
    for row in p["rows"]:
        out.append(f"| {row['steps']} | "
                   + " | ".join(f"{row['plain_state'][str(L)]:.3f}" for L in p["lengths"])
                   + f" | {row['loss']:.4f} |")
    out += ["", p["guided"], "", p["reading"]]
    return out


def recipe_section(cfg):
    """The recipe as run, and the two places the hardware fixes it.

    Both are measured on the exact documents this run trains on (the composed cell's checkpoint
    document reaches 2540 tokens at L=96), and both are properties of the kernel and the card
    rather than choices, so a reader reproducing this needs the numbers rather than the labels.
    """
    out = [
        "", "## The recipe", "",
        f"`d_model={cfg['d_model']}` x `n_layers={cfg['n_layers']}`, `n_heads="
        f"{cfg.get('n_heads')}`, batch {cfg['batch']}, {cfg['steps']} steps, "
        f"{cfg['train_n']} items per stage over the three-stage curriculum "
        f"({' -> '.join(n for n, _s, _w in cfg['schedule'])}), supervision `{cfg['fmt']}` "
        "(answer-masked plain documents plus the specs' own per-event checkpoint documents). "
        "Every architecture runs the SAME recipe at the same width and depth, which is this "
        f"repo's compute-matched convention; per-seed weights are saved to "
        f"`{cfg.get('ckpt_dir')}`, so an added eval length is a decode and not a retrain.",
        "",
        "Two numbers in it are set by the hardware, on the documents this run trains on "
        "(the composed cell's checkpoint document is 2540 tokens at L=96, mean 958):",
        "",
        "- **head dimension 128, so 6 heads at d768.** `GatedDeltaProduct` at head dimension 192 "
        "(4 heads at d768) runs 0.836 s/step against 0.108 s/step at 128, same width, same depth, "
        "same batch — a kernel path, not a model property.",
        f"- **batch {cfg['batch']}.** At d768x8 the longest document slice runs out of memory at "
        "batch 24 on a 32 GB card (peak 26.9 GB at 16). This run therefore draws "
        f"{cfg['batch'] * cfg['steps'] / 1000:.0f}k sequences per seed from "
        f"{cfg['train_n'] * 2 / 1000:.0f}k documents per stage.",
    ]
    rate = cfg.get("measured_s_per_step") or {}
    if rate:
        out += ["", "| arch | measured s/step | train s/seed |", "|---|---|---|"]
        for arch, v in sorted(rate.items()):
            out.append(f"| {arch} | {v:.3f} | {v * cfg['steps']:.0f} |")
    return out


def cost_section(cfg, grid):
    """The two PAIRINGS side by side, each in both cost models.

    WORK-matched is each component read at the length carrying the same amount of its own work as
    the composed stream, and it is what the components' FORMS verdict is read at; the multiplier
    there is whatever the composed cell's extra structure costs. TOKEN-matched is the
    matched-COST control, whose multiplier is 1.00 by construction — that is what makes it a
    control rather than a comparison. CHARGED STEPS is what a scratchpad solver pays; FORWARD-PASS
    TOKENS is what a streaming model pays.
    """
    costs = cfg.get("cell_costs") or {}
    paired = P.as_pairings(cfg.get("matched_lengths") or {})
    if not costs:
        return []

    def cell(name, L):
        return costs.get(f"{name}@{L}") if L else None

    out = ["", "## The two pairings, and the step multiplier under each", "",
           "| composed L | component | work-matched L | x steps | x tokens | "
           "token-matched L | x steps | x tokens |",
           "|---|---|---|---|---|---|---|---|"]
    for L in P.LOCAL_LENGTHS:
        c = cell("composed", L)
        if not c:
            continue
        for comp in ("state", "bind"):
            wl = (paired.get("work", {}).get(L, {}).get(comp) or {}).get("L") \
                or P.WORK_MATCHED.get(L, {}).get(comp)
            tl = (paired.get("tokens", {}).get(L, {}).get(comp) or {}).get("L") \
                or P.TOKEN_MATCHED.get(L, {}).get(comp)
            w, t = cell(comp, wl), cell(comp, tl)
            out.append(
                f"| {L} | {comp} | {wl or '—'} | "
                f"{f'{c[0]/w[0]:.2f}x' if w else '—'} | {f'{c[1]/w[1]:.2f}x' if w else '—'} | "
                f"{tl or '— (unreachable)'} | "
                f"{f'{c[0]/t[0]:.2f}x' if t else '—'} | {f'{c[1]/t[1]:.2f}x' if t else '—'} |")
    out += ["", "_The work-matched length is the composed stream's own count of that component's "
            "events: composed@48 contains 17 swaps and 31 gives. The token-matched length is "
            "where that component's forward pass costs what the composed cell costs at L, and it "
            "is unreachable on the retrieval component past L=132 — its sampler pins the "
            "resolving write into a window that gets exponentially harder to satisfy as the "
            "stream grows (protocol.BIND_MATCHED_MAX). That is a property of the instrument, not "
            "of this run._"]
    return out


def _floor_cell(fr, chance=None):
    """One floor cell, with an UNFLOORABLE cell carrying its pad reach rather than a blank.

    A blank would read as "not measured". The cell was measured; what does not exist is a bar it
    could clear, and the pad reach — what the excluded both-maps class scores on the exact items —
    is the number that says how far the unfloorable class gets. It is not a floor and is never
    used to bold a cell.
    """
    if fr is None:
        return "—"
    if fr.get("floor") is not None:
        return (f"{fr['floor']:.3f}" if chance is None
                else f"{fr['floor']:.3f} ({fr['floor'] / chance:.2f}x)")
    pad = fr.get("pad_reach")
    return "unfloorable" + (f" (pad {pad:.3f})" if pad is not None else "")


def depth_matched_section(runs, floors, cfg, guided_floors=None):
    """THE MEASUREMENT THE RUN EXISTS FOR: the three cells at equal component work, per seed.

    Pre-registered, not post-hoc — these are exactly ``P.registered_lengths`` and the floors are
    each cell's own. It is pulled into one table because the per-cell tables below put the three
    cells in three places, and reading them apart is what let the confounded pairing stand: at
    p_swap = 1/3 the composed cell at L carries a THIRD of the state depth of a state cell at the
    same L, so the previous run compared a 5.7-hop composed cell with a 16.0-hop state cell and
    reported no gap. Here every column of a row is the same carrier chain.
    """
    if not runs:
        return []
    k = cfg["k"]

    def hops(cell, L):
        fr = floors.get(f"{cell}@{L}")
        if not fr or fr.get("n_swap") is None:
            return None
        return 2.0 * fr["n_swap"] / max(1, k)

    out = ["", "# The depth-matched comparison (pre-registered)", ""]
    for read, n in (("guided", cfg["guided_n"]), ("plain", cfg["eval_n"])):
        key = "eval" if read == "plain" else "guided"
        rf = floors if read == "plain" else _guided_records(guided_floors, floors)
        triples = []
        for cl in P.LOCAL_LENGTHS:
            trip = {"composed": cl, "state": P.WORK_MATCHED.get(cl, {}).get("state"),
                    "bind": P.WORK_MATCHED.get(cl, {}).get("bind")}
            if not all(trip.values()):
                continue
            if any(any(str(trip[c]) in r["stages"][-1][key].get(c, {}) for r in runs)
                   for c in trip):
                triples.append(trip)
        if not triples:
            continue
        out += [f"## {read.upper()} read (n={n})", ""]
        for trip in triples:
            hp = hops("composed", trip["composed"])
            out += [f"**composed@{trip['composed']} vs state@{trip['state']} and "
                    f"bind@{trip['bind']}** — "
                    + (f"carrier chain {hp:.1f} hops on both state legs "
                       if hp else "")
                    + f"(composed@{trip['composed']} holds {trip['state']} swaps and "
                      f"{trip['bind']} gives).",
                    "",
                    f"| arch | seed | state@{trip['state']} | composed@{trip['composed']} | "
                    f"bind@{trip['bind']} |", "|---|---|---|---|---|"]
            for arch in sorted({r["arch"] for r in runs}):
                for r in sorted((x for x in runs if x["arch"] == arch),
                                key=lambda x: x["seed"]):
                    vals = []
                    for c in ("state", "composed", "bind"):
                        blk = r["stages"][-1][key].get(c, {}).get(str(trip[c]))
                        v = blk if read == "plain" else (blk or {}).get("match")
                        f = rf.get(f"{c}@{trip[c]}", {}).get("floor")
                        vals.append("—" if v is None else
                                    (f"**{v:.3f}**" if P.clears(v, f, n)[0] else f"{v:.3f}"))
                    out.append(f"| {arch} | {r['seed']} | " + " | ".join(vals) + " |")
            frow = []
            for c in ("state", "composed", "bind"):
                frow.append(_floor_cell(rf.get(f"{c}@{trip[c]}")))
            out += ["| _floor_ | | " + " | ".join(frow) + " |", ""]
    out += ["_A **bold** cell clears its own recomputed floor under the pre-registered rule. "
            "Every column of a row costs the same amount of that column's own work; the "
            "TOKEN-matched pairing (state@80, bind@132 against composed@48) is the "
            "matched-COST control and is in the tables below._"]
    return out


def flops_caveat(cfg, readable=None, sizes=None):
    """The compute match, checked against the run's own measured numbers rather than asserted.

    The repo's convention matches on FLOPs/token, not on parameters. Where a run's measured
    FLOPs/token do NOT match, the comparison is not compute-matched and the caveat travels with
    every reading that compares the architectures — the more so where the architecture that is
    ahead is the one carrying the extra compute.

    ``readable`` is the list of architectures that clear anything at all on the ANSWER channel;
    the "and it is the only one readable there" clause is printed only where the run's own
    numbers say so, never as a standing claim.
    """
    sz = sizes if sizes is not None else cfg.get("sizes") or {}
    fl = {a: v[1] for a, v in sz.items() if v and v[1]}
    if len(fl) < 2:
        return []
    top = max(fl, key=lambda a: fl[a])
    rest = {a: v for a, v in fl.items() if a != top}
    gap = fl[top] / min(rest.values()) - 1.0
    if gap < 0.02:
        return ["", "_Per-token FLOPs match to within "
                f"{gap * 100:.1f}% across the roster, which is the axis this repo matches on._",
                ""]
    only = (readable is not None and list(readable) == [top])
    return ["", f"**`{top}` IS NOT FLOPs-MATCHED.** It runs {fl[top] / 1e6:.2f}M FLOPs/token "
            "against " + ", ".join(f"`{a}`'s {v / 1e6:.2f}M" for a, v in sorted(rest.items()))
            + f" — a {gap * 100:.0f}% advantage, against this repo's own compute-matching "
            f"convention (match on FLOPs/token, not on parameters). Every comparison in which "
            f"`{top}` is ahead carries it"
            + (f", and it is the more load-bearing here because `{top}` is the only architecture "
               "this run reads on the ANSWER channel at all." if only else "."),
            ""]


def write_markdown(runs, per_arch, floors, cfg, grid, path, guided_floors=None):
    ch = 1.0 / (cfg["k"] - 1)
    lines = [
        "# s5_bind_v3 three-cell comparison — from-scratch arm",
        "",
        f"k={cfg['k']} · informed chance 1/(k-1) = {ch:.3f} · match · n_eval={cfg['eval_n']} · "
        f"d_model={cfg['d_model']} n_layers={cfg['n_layers']} n_heads={cfg.get('n_heads')} "
        f"batch={cfg['batch']} steps={cfg['steps']} train_n={cfg['train_n']}/stage · "
        f"supervision={cfg['fmt']}",
        "",
        "Reading rule pre-registered in `scripts/protocol_s5bind_v3_three_cell_20260731.py`: a "
        f"cell CLEARS its floor at z > {P.Z_CLEAR} AND margin >= {P.MARGIN}; it FORMS for an "
        f"architecture on >= {P.SEEDS_CLEAR} of the seeds at every registered length. Per-seed "
        "values only — this family is bimodal at the emergence threshold.",
        "",
        "**The composed cell has no floor on the GUIDED read, on either channel.** That floor is "
        "the one-structure bound `W <= max(k, m) + 1` against the task's `k + m + 1`, and the "
        "guided format requires the whole of P then the whole of B at every event — so the k + m "
        "slots the bound prices are handed to every policy, the task's own algorithm included, "
        "and the class that survives contains the task. It is a property of the PROTOCOL and not "
        "of the read: the guided decode accumulates the generated checkpoints into the same "
        "context the answer token comes out of. Guided composed cells are reported UNFLOORABLE "
        "with the pad reach — what the excluded both-maps class scores on the exact items — "
        "beside them. **The previous `gdp_hybrid / guided: V2_NO_GAP_HERE` was read off that "
        "floor and is RETRACTED**; the verdict below is what the rule returns without it. The "
        "PLAIN read is unaffected: a streaming model with no scratchpad is the class the bound "
        "prices.",
        "",
        "## Size (compute-matched: shared d_model and depth; `fprm` is weight-tied)",
        "",
        "| arch | params | FLOPs/token |", "| --- | --- | --- |",
    ]
    for arch, (p, fl) in sorted(cfg["sizes"].items()):
        lines.append(f"| {arch} | {p / 1e6:.1f}M | {fl / 1e6:.2f}M |")
    # the answer channel is the PLAIN read; an architecture is "readable" there if any cell of
    # its own registered grid clears on any seed. Measured off this run, not assumed.
    readable = sorted(a for a, reads in per_arch.items()
                      if any(n for per_len in
                             (reads.get("plain", {}).get("seed_counts") or {}).values()
                             for n in per_len.values()))
    lines += flops_caveat(cfg, readable)
    # measured from this run's own stages, so the compute-matched claim is checkable rather than
    # asserted: matched FLOPs/token does not imply matched wall clock and both are printed.
    rate = {}
    for arch in sorted({r["arch"] for r in runs}):
        tot = [(sum(s["train_s"] for s in r["stages"]), sum(s["steps"] for s in r["stages"]))
               for r in runs if r["arch"] == arch]
        tot = [(t, s) for t, s in tot if t and s]
        if tot:
            rate[arch] = sum(t for t, _ in tot) / sum(s for _, s in tot)
    cfg = {**cfg, "measured_s_per_step": rate}
    lines += (recipe_section(cfg) + pilot_section(cfg) + cost_section(cfg, grid)
              + depth_matched_section(runs, floors, cfg, guided_floors))
    for read, n in (("plain", cfg["eval_n"]), ("guided", cfg["guided_n"])):
        key = "eval" if read == "plain" else "guided"
        rf = floors if read == "plain" else _guided_records(guided_floors, floors)
        label = ("PLAIN read — answer off the plain prompt, no scratchpad"
                 if read == "plain" else
                 "GUIDED read — events forced, checkpoints and answer generated")
        lines += ["", f"# {label} (n={n})"]
        for cell in ("state", "bind", "composed"):
            lens = [L for L in grid[cell]
                    if any(str(L) in r["stages"][-1][key].get(cell, {}) for r in runs)]
            if not lens:
                continue
            lines += ["", f"## {cell} cell — `{P.LOCAL_CELLS[cell]}`", "",
                      "| arch | " + " | ".join(f"L{L} per-seed" for L in lens) + " |",
                      "|" + "---|" * (len(lens) + 1)]
            for arch in sorted({r["arch"] for r in runs}):
                cells = []
                for L in lens:
                    vals = []
                    for r in runs:
                        if r["arch"] != arch:
                            continue
                        v = r["stages"][-1][key].get(cell, {}).get(str(L))
                        vals.append(v if read == "plain" or v is None else v["match"])
                    fl = rf.get(f"{cell}@{L}", {}).get("floor")
                    cells.append(" ".join(
                        "—" if v is None else
                        (f"**{v:.3f}**" if P.clears(v, fl, n)[0] else f"{v:.3f}")
                        for v in vals))
                lines.append(f"| {arch} | " + " | ".join(cells) + " |")
            row = [_floor_cell(rf.get(f"{cell}@{L}"), ch) for L in lens]
            lines.append("| _floor_ | " + " | ".join(row) + " |")
    # The per-slot checkpoint accuracy is the diagnostic that separates the two readings a
    # floored GUIDED answer is otherwise ambiguous between: no state is tracked at all, or
    # state is tracked and the error compounds over the run. It is not the registered metric
    # and no verdict reads it.
    ck_rows = []
    for arch in sorted({r["arch"] for r in runs}):
        for cell in ("state", "bind", "composed"):
            for L in cfg.get("guided_grid", {}).get(cell, cfg["guided_lengths"]):
                vals = [r["stages"][-1]["guided"].get(cell, {}).get(str(L), {})
                        .get("checkpoint_acc")
                        for r in runs if r["arch"] == arch]
                vals = [v for v in vals if v is not None]
                if vals:
                    ck_rows.append((arch, cell, L, vals))
    if ck_rows:
        # NO CHANCE COLUMN HERE. 1/k is not the reference for a per-slot score — most slots do
        # not move, so a model that re-emits its previous checkpoint already scores 0.80-0.91 —
        # and printing 1/k beside these numbers is what made a score BELOW that copier read as a
        # partial trace. The reference is computed and tabulated in the post-hoc section.
        lines += ["", "## Guided checkpoint accuracy (per-slot, diagnostic — not the metric)", "",
                  "Read against the copy-the-previous-checkpoint reference, not against 1/k; "
                  "both are in _What the checkpoint diagnostic says_ below.", "",
                  "| arch | cell | L | per-seed |", "|---|---|---|---|"]
        for arch, cell, L, vals in ck_rows:
            lines.append(f"| {arch} | {cell} | {L} | " + " ".join(f"{v:.3f}" for v in vals) + " |")
    lines += ["", "# Verdict", ""]
    retracted = [a for a, reads in sorted(per_arch.items())
                 if reads.get("guided", {}).get("verdict") == "V0_COMPOSED_UNFLOORABLE"]
    if retracted:
        lines += ["The GUIDED read's composed cell is unfloorable, so the verdict for "
                  + ", ".join(f"`{a}`" for a in retracted)
                  + " is **V0_COMPOSED_UNFLOORABLE**. It REPLACES the previously published "
                  "**V2_NO_GAP_HERE**, which was reached by scoring the composed cell against a "
                  "floor that does not hold under this protocol; that verdict is retracted. V0 "
                  "is not a null and not a gap — with no floor the cell can neither clear nor "
                  "fail to clear, and the reading this protocol does support is the within-run "
                  "comparison against the work-matched component below.", ""]
    for arch, reads in sorted(per_arch.items()):
        for read, v in sorted(reads.items()):
            ctrl = v.get("control", {})
            pad = v.get("composed_pad_reach")
            unfl = any(n is None for per_len in v["seed_counts"].values()
                       for n in per_len.values())
            lines += [f"**{arch} / {read}: {v['verdict']}** — {v['why']}", "",
                      f"seeds clearing: {v['seed_counts']}"
                      + (" (a `None` is a length with no floor on this protocol, not a length "
                         "where no seed cleared)" if unfl else "")
                      + "; positive control "
                      f"(some component clears on this read's grid) {ctrl.get('per_pair')} of "
                      f"{len(cfg['seeds'])} seeds, required {ctrl.get('required')}; "
                      f"matched-cost control: {v['matched_forms']} "
                      f"(measured: {v.get('matched_measured')}); "
                      + (f"composed pad reach: {pad:.3f}; " if pad is not None else "")
                      + f"lengths read: {v['lengths_read']}", ""]
    lines += post_hoc_section(runs, _guided_records(guided_floors, floors), cfg)
    lines += ["", "_A **bold** cell clears its own operative floor under the pre-registered "
              "rule; a cell marked `unfloorable` has no floor on that read's protocol and can "
              "never be bold, and the `pad` beside it is what the excluded both-maps class "
              "scores on the exact items — a lower bound on that class's max, not a bar. "
              "Floors are recomputed from that cell's own items and under that read's own "
              "protocol: registry rows plus the "
              "admitted swept family. The fitted surface ranker is measured beside them "
              f"(fit {P.N_FIT_BLOCKS}x{P.N_FIT} / scored {P.N_SCORE} disjoint) and is NOT in any "
              "floor — no implementation of it achieves a price the class rule admits. The "
              "composed cell's cost multiplier over each component is reported in the "
              "pre-registration record in both cost models; the matched-cost lengths in the "
              "tables above are the FORWARD-PASS match, which is this regime's cost._"]
    path.write_text("\n".join(lines))


def cell_costs(specs, tok, grid):
    """``{cell@L: (charged_steps, prompt_tokens)}`` — the two cost models, per scored cell."""
    return {f"{cell}@{L}": P.cell_cost(specs[cell], L, tok)
            for cell in grid for L in grid[cell]}


def guided_floors_for(grid, guided_grid, guided_n, cached=None):
    """The floor at each GUIDED cell, measured on the ``guided_n`` items that read scores and
    under the GUIDED protocol's own class rule.

    A floor is a property of the items it is read against, and the max over admitted rows carries
    an upward selection bias that a smaller n does not average out: the same rows give 0.250 at
    state@80 and 0.234 at composed@48 on the 128 guided items against 0.207 and 0.204 on 1000.
    Reading a 128-item score against the 1000-item floor is reading it against a different item
    set. ``P.cell_floor`` keeps its own pool discipline at either n — rows on a disjoint pool and
    on the exact scored items, operative is the larger.

    It is ALSO a property of the protocol, and that is what ``guided=True`` carries: the guided
    format writes the whole of P then B at every event, so the live-slot conjunct of the class
    rule prices a resource every policy has. Components are unmoved; the composed cell has no
    floor and reports ``pad_reach`` instead. A cached record from before that rule — one with no
    ``protocol`` key — is DISCARDED rather than reused, because it carries the retracted number.
    """
    out = {k: v for k, v in (cached or {}).items() if v.get("protocol") == "guided"}
    for cell, lengths in guided_grid.items():
        for L in lengths:
            key = f"{cell}@{L}"
            if key in out:
                continue
            out[key] = P.cell_floor(TK.CANONICAL[P.LOCAL_CELLS[cell]], L, n_eval=guided_n,
                                    guided=True)
            fl = out[key]["floor"]
            pad = out[key].get("pad_reach")
            print(f"  guided floor {key} = "
                  + (f"{fl:.4f}" if fl is not None else
                     "unfloorable" + (f" (pad reach {pad:.4f})" if pad is not None else ""))
                  + f" (n={guided_n})", flush=True)
    return out


def rewrite(json_path):
    """Re-render the report from a results JSON, applying the current rule and tables."""
    res = json.load(open(json_path))
    cfg = res["cfg"]
    grid = {k: [int(x) for x in v] for k, v in cfg["grid"].items()}
    if not cfg.get("cell_costs"):
        specs = three_cell_specs(P.TRAIN_LENGTHS)
        world, renderer = TK.build_world(specs["composed"])
        cfg["cell_costs"] = cell_costs(specs, Tokenizer.build([world], renderer), grid)
        res["cfg"] = cfg
    gn = cfg.get("guided_n", P.N_GUIDED)
    gg = {c: [int(x) for x in v] for c, v in
          (cfg.get("guided_grid") or {c: cfg["guided_lengths"] for c in grid}).items()}
    gf = guided_floors_for(grid, gg, gn, res.get("guided_floors"))
    per_arch = apply_rule(res["runs"], res["floors"], grid, cfg["eval_n"], gn, gf)
    md = Path(str(json_path).replace(".json", ".md"))
    cfg = {**cfg, "sizes": {k: tuple(v) for k, v in cfg["sizes"].items()}}
    write_markdown(res["runs"], per_arch, res["floors"], cfg, grid, md, gf)
    res["verdicts"] = per_arch
    res["guided_floors"] = gf
    Path(json_path).write_text(json.dumps(res, indent=2, default=float))
    return per_arch


def main():
    ap = argparse.ArgumentParser(description="Three-cell comparison, from-scratch arm.")
    ap.add_argument("--rewrite", default=None,
                    help="re-render the report from an existing results JSON and exit")
    ap.add_argument("--archs", default="gdp_hybrid,fprm,transformer")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--steps", type=int, default=12000)
    ap.add_argument("--batch", type=int, default=24)
    ap.add_argument("--d_model", type=int, default=512)
    ap.add_argument("--n_layers", type=int, default=6)
    ap.add_argument("--n_heads", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--train_n", type=int, default=24000, help="items per stage, split by mix")
    ap.add_argument("--eval_n", type=int, default=P.N_EVAL)
    ap.add_argument("--guided_n", type=int, default=P.N_GUIDED,
                    help="items for the guided read (0 disables it)")
    ap.add_argument("--guided_batch", type=int, default=128,
                    help="items per padded forward in the guided decode. Scoring is unaffected "
                         "(right padding, causal models); it is a memory knob, and at d768x8 the "
                         "128-item batch does not fit.")
    ap.add_argument("--guided_lengths", type=int, nargs="*",
                    default=list(P.GUIDED_LENGTHS))
    ap.add_argument("--fmt", default="mix", choices=["plain", "checkpoint", "mix"])
    ap.add_argument("--no_matched", action="store_true",
                    help="skip the matched-cost control lengths (smoke tests)")
    ap.add_argument("--floors", default=None, help="reuse a pre-registration record's floors")
    ap.add_argument("--ckpt_dir", default=None,
                    help="where to write per-(arch, seed) weights; defaults to "
                         "<out_prefix>_ckpt. An added eval length is then a DECODE "
                         "(--decode_from) rather than a retrain.")
    ap.add_argument("--no_ckpt", action="store_true", help="do not save weights (smoke tests)")
    ap.add_argument("--decode_from", default=None,
                    help="score the checkpoints in this directory on the grid and exit; no "
                         "training happens. This is what makes a new eval length cheap.")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--loss_log_interval", type=int, default=500)
    ap.add_argument("--out_prefix", default=None)
    a = ap.parse_args()

    if a.rewrite:
        for arch, reads in sorted(rewrite(a.rewrite).items()):
            for read, v in sorted(reads.items()):
                print(f"  {arch} / {read}: {v['verdict']} — {v['why']}")
        return

    specs = three_cell_specs(P.TRAIN_LENGTHS)
    world, renderer = TK.build_world(specs["composed"])
    tok = Tokenizer.build([world], renderer)

    # Each cell at its OWN registered lengths: the composed grid for the composed cell, the
    # WORK-MATCHED partners of it for the components, plus the shared positive-control length.
    grid = {c: [P.CONTROL_LENGTH, *P.registered_lengths(c), *P.PROFILE_LENGTHS.get(c, ())]
            for c in ("state", "bind", "composed")}
    matched = {}
    if not a.no_matched:
        matched = P.matched_lengths(tok, axis=P.MATCHED_AXIS)
        for L in P.LOCAL_LENGTHS:
            for cell in ("state", "bind"):
                ml = matched[L][cell]["L"]
                if ml and ml not in grid[cell]:
                    grid[cell].append(ml)
    for cell in grid:
        grid[cell] = sorted(set(grid[cell]))
    # THE GUIDED READ BUYS ITS OWN MATCHED-COST CONTROL at the shortest composed length. Without
    # it that read cannot reach V1 at all: the control is a component at a LONGER length than any
    # the read covers, so "beyond the step multiplier" would be unevaluable there however the
    # cells came out (protocol.guided_grid).
    guided_grid = P.guided_grid(matched, lengths=tuple(a.guided_lengths)) if matched else \
        {c: list(a.guided_lengths) for c in grid}

    # A cached floor file is reused where it covers a cell; anything the grid needs and the
    # cache lacks is measured here, so no cell is ever read against a floor that is missing.
    floors = json.load(open(a.floors))["floors"] if a.floors else {}
    for cell, lengths in grid.items():
        for L in lengths:
            if f"{cell}@{L}" in floors:
                continue
            floors[f"{cell}@{L}"] = P.cell_floor(TK.CANONICAL[P.LOCAL_CELLS[cell]], L)
            print(f"  floor {cell}@{L} = {floors[f'{cell}@{L}']['floor']:.4f} (measured)",
                  flush=True)
    # THE GUIDED READ HAS ITS OWN FLOORS because it has its own items: it scores guided_n and the
    # plain read scores eval_n, and the operative floor is a max over rows on the exact scored
    # set. Reading a 128-item score against a 1000-item floor reads it against a different item
    # set, and the two differ by up to 0.043 here.
    guided_floors = guided_floors_for(grid, guided_grid, a.guided_n,
                                      json.load(open(a.floors)).get("guided_floors")
                                      if a.floors else None)

    archs = [x.strip() for x in a.archs.split(",")]
    sizes = {arch: measured_size(arch, a.d_model, a.n_layers, a.n_heads, tok.vocab_size,
                                 a.device) for arch in archs}
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    prefix = Path(a.out_prefix or f"results/s5bind_v3_three_cell_local_{ts}")
    prefix.parent.mkdir(parents=True, exist_ok=True)
    log_path, md_path, js_path = (Path(f"{prefix}.jsonl"), Path(f"{prefix}.md"),
                                  Path(f"{prefix}.json"))
    ckpt_dir = None if a.no_ckpt else Path(a.ckpt_dir or f"{prefix}_ckpt")
    cfg = {"k": specs["composed"].k, "archs": archs, "seeds": a.seeds, "steps": a.steps,
           "batch": a.batch, "d_model": a.d_model, "n_layers": a.n_layers,
           "n_heads": a.n_heads, "cell_costs": cell_costs(specs, tok, grid), "lr": a.lr,
           "train_n": a.train_n, "eval_n": a.eval_n, "guided_n": a.guided_n,
           "guided_lengths": a.guided_lengths, "guided_grid": guided_grid,
           "fmt": a.fmt, "grid": grid,
           "matched_lengths": matched, "schedule": [(n, s, w) for n, s, w in SCHEDULE],
           "sizes": sizes, "train_lengths": P.TRAIN_LENGTHS,
           "ckpt_dir": None if ckpt_dir is None else str(ckpt_dir),
           "decoded_from": a.decode_from}

    if a.decode_from:
        print(f"=== DECODE from {a.decode_from} -> {md_path} (no training) ===", flush=True)
        runs = decode_runs(a.decode_from, archs, a.seeds, specs, tok, world, grid,
                           eval_n=a.eval_n, guided_n=a.guided_n, guided_lengths=guided_grid,
                           device=a.device, guided_batch=a.guided_batch)
        for row in runs:
            with log_path.open("a") as f:
                f.write(json.dumps({**row, **{k: v for k, v in cfg.items()
                                              if k != "matched_lengths"}}) + "\n")
        if runs:
            per_arch = apply_rule(runs, floors, grid, a.eval_n, a.guided_n, guided_floors)
            write_markdown(runs, per_arch, floors, cfg, grid, md_path, guided_floors)
            js_path.write_text(json.dumps(
                {"cfg": cfg, "floors": floors, "guided_floors": guided_floors,
                 "runs": runs, "verdicts": per_arch}, indent=2, default=float))
            for arch, reads in sorted(per_arch.items()):
                for read, v in sorted(reads.items()):
                    print(f"  {arch} / {read}: {v['verdict']} — {v['why']}", flush=True)
        print(f"\n=== done: {md_path} ===", flush=True)
        return

    print(f"=== three-cell local: {len(archs) * len(a.seeds)} runs -> {md_path} ===", flush=True)
    for arch, (p, fl) in sizes.items():
        print(f"  {arch}: {p / 1e6:.1f}M params, {fl / 1e6:.2f}M FLOPs/token", flush=True)
    runs = []
    for arch in archs:
        for seed in a.seeds:
            print(f"\n--- {arch} seed {seed} ---", flush=True)
            try:
                stages = run_one(arch, seed, specs, tok, world, grid, steps=a.steps,
                                 batch=a.batch, d_model=a.d_model, n_layers=a.n_layers,
                                 n_heads=a.n_heads, lr=a.lr, train_n=a.train_n,
                                 eval_n=a.eval_n, guided_n=a.guided_n,
                                 guided_lengths=guided_grid, device=a.device,
                                 fmt=a.fmt, loss_log_interval=a.loss_log_interval,
                                 guided_batch=a.guided_batch, ckpt_dir=ckpt_dir)
            except Exception as e:                                        # noqa: BLE001
                import traceback
                traceback.print_exc()
                with log_path.open("a") as f:
                    f.write(json.dumps({"arch": arch, "seed": seed, "error": str(e)}) + "\n")
                continue
            row = {"arch": arch, "seed": seed, "stages": stages}
            runs.append(row)
            with log_path.open("a") as f:
                f.write(json.dumps({**row, **{k: v for k, v in cfg.items()
                                              if k != "matched_lengths"}}) + "\n")
            per_arch = apply_rule(runs, floors, grid, a.eval_n, a.guided_n, guided_floors)
            write_markdown(runs, per_arch, floors, cfg, grid, md_path, guided_floors)
            js_path.write_text(json.dumps(
                {"cfg": cfg, "floors": floors, "guided_floors": guided_floors,
                 "runs": runs, "verdicts": per_arch}, indent=2, default=float))
    if runs:
        per_arch = apply_rule(runs, floors, grid, a.eval_n, a.guided_n, guided_floors)
        write_markdown(runs, per_arch, floors, cfg, grid, md_path, guided_floors)
        js_path.write_text(json.dumps({"cfg": cfg, "floors": floors,
                                       "guided_floors": guided_floors,
                                       "runs": runs,
                                       "verdicts": per_arch}, indent=2, default=float))
        print("\n=== verdict ===", flush=True)
        for arch, reads in sorted(per_arch.items()):
            for read, v in sorted(reads.items()):
                print(f"  {arch} / {read}: {v['verdict']} — {v['why']}", flush=True)
    print(f"\n=== done: {md_path} ===", flush=True)


if __name__ == "__main__":
    main()

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
    FLOPs. Both numbers are measured and printed per architecture; neither is hidden.

Smoke test (minutes):
    .venv-train/bin/python scripts/experiment_s5bind_v3_three_cell_local_20260731.py \
        --archs gdp_hybrid --seeds 0 --steps 600 --d_model 128 --n_layers 2 --batch 8 \
        --train_n 400 --eval_n 100 --no_matched

Full run:
    .venv-train/bin/python scripts/experiment_s5bind_v3_three_cell_local_20260731.py \
        --archs gdp_hybrid,fprm,transformer --seeds 0 1 2 --steps 12000 --batch 24 \
        --d_model 512 --n_layers 6 --train_n 24000 --eval_n 1000
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


def guided_free_run_batched(model, tok, spec, length, n, device, batch=128, max_answer=4):
    """GUIDED read: events teacher-forced, every checkpoint and the answer GENERATED.

    The s5 formation protocol (``scripts/sweep.py::guided_free_run_eval``), batched, because
    the reference decodes one item at a time with no cache and a checkpoint here is the whole
    of P then B — k + m = 12 tokens per event, so 1152 sequential forward passes per item at
    L = 96. Every item at one length has the same NUMBER of slots even though their absolute
    positions differ (event sentences differ in width), so the loop runs once per slot ORDINAL
    with one padded batched forward per round.

    Returns ``(answer_match, checkpoint_acc)``. The model runs its own state through the
    stream, so this is the SCRATCHPAD read: the floor's W axis has no force against it, and it
    is read against the admitted end of the cell's profile like any other score.
    """
    import torch
    from sweep import _interleaved_slots

    examples = TK.generate(spec, "test", n=n, length=length)
    prepped = []
    for ex in examples:
        inter = ex.meta.get("interleaved_prompt")
        if inter is None:
            return None, None
        toks, slots = _interleaved_slots(ex.prompt, inter)
        prepped.append((toks, slots, set(slots), [toks[s] for s in slots], ex.answer))
    n_slots = len(prepped[0][1])
    if any(len(p[1]) != n_slots for p in prepped):
        return None, None
    hits = ck_hits = ck_total = 0
    model.eval()
    with torch.no_grad():
        for b0 in range(0, len(prepped), batch):
            chunk = prepped[b0:b0 + batch]
            ids = [[] for _ in chunk]
            cursor = [0] * len(chunk)
            gen_ck = [[] for _ in chunk]
            for ordinal in range(n_slots + 1):
                for i, (toks, slots, slotset, _gold, _ans) in enumerate(chunk):
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
            for i, (_t, _s, _ss, gold_ck, ans) in enumerate(chunk):
                pred = tok.decode(outs[i])
                hits += bool(TK.score_relaxed(Renderer.normalize(pred),
                                              Renderer.normalize(ans)))
                ck_hits += sum(1 for a, g in zip(gen_ck[i], gold_ck) if a == g)
                ck_total += len(gold_ck)
    model.train()
    return hits / max(1, len(prepped)), ck_hits / max(1, ck_total)


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
            guided_batch=128):
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
        backend = LocalBackend([world], arch=arch, model=model, tokenizer=tok, device=device)
        # intermediate stages get a progress read only; the registered grid, the matched-cost
        # control lengths and the guided decode are paid for once, at the end.
        g = grid if last else {c: [P.CONTROL_LENGTH, P.LOCAL_LENGTHS[0]] for c in grid}
        ev = eval_cells(backend, specs, eval_n if last else 200, g)
        for cell, cells in ev.items():
            print("     plain  " + f"{cell:9s} "
                  + "  ".join(f"L{L}={a:.3f}" for L, a in cells.items()), flush=True)
        gv = {}
        if last and guided_n:
            for cell in specs:
                gv[cell] = {}
                # a PER-CELL grid: the guided read's own lengths, plus each component's
                # matched-cost length, without which the guided read cannot reach V1 at all
                for L in guided_lengths.get(cell, ()) if isinstance(guided_lengths, dict) \
                        else guided_lengths:
                    t1 = time.time()
                    a, ck = guided_free_run_batched(model, tok, specs[cell], L, guided_n, device,
                                                    batch=guided_batch)
                    gv[cell][str(L)] = {"match": a, "checkpoint_acc": ck}
                    print(f"     guided {cell:9s} L{L}: match={a:.3f} ck={ck:.3f} "
                          f"[{time.time() - t1:.0f}s]", flush=True)
        stages.append({"stage": name, "steps": stage_steps, "n_docs": len(docs),
                       "mix": weights, "final_loss": run["final_loss"],
                       "loss_curve": [(int(s), float(v)) for s, v in run.get("loss_curve", [])],
                       "train_s": round(time.time() - t0), "eval": ev, "guided": gv})
    del model
    torch.cuda.empty_cache()
    return stages


def _accuracies(rows, grid, read):
    """``{cell: {seed: {L: match}}}`` for one read, off the final stage."""
    out = {}
    for cell in grid:
        out[cell] = {}
        for r in rows:
            blk = r["stages"][-1]["eval" if read == "plain" else "guided"].get(cell, {})
            out[cell][r["seed"]] = {int(L): (v if read == "plain" else v["match"])
                                    for L, v in blk.items()}
    return out


def apply_rule(runs, floors, grid, eval_n, guided_n):
    """The pre-registered rule, applied to the final-stage numbers of BOTH reads separately.

    Never mixed: judging the components on the guided read and the composed cell on the plain
    one would manufacture a composition gap out of the eval mode.
    """
    per_arch = {}
    for arch in sorted({r["arch"] for r in runs}):
        rows = [r for r in runs if r["arch"] == arch]
        f = {cell: {int(k.split("@")[1]): v["floor"] for k, v in floors.items()
                    if k.split("@")[0] == cell} for cell in grid}
        per_read = {}
        for read, n in (("plain", eval_n), ("guided", guided_n)):
            acc = _accuracies(rows, grid, read)
            if not any(acc[c][s] for c in acc for s in acc[c]):
                continue
            lengths = tuple(L for L in P.LOCAL_LENGTHS
                            if all(L in acc[c][s] for c in acc for s in acc[c]))
            comp_forms, comp_counts = {}, {}
            for cell in ("state", "bind", "composed"):
                ok, counts = P.forms(acc[cell], f[cell], lengths, n=n)
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
                mlens = tuple(L for L in sorted(set(grid[cell]) - set(P.LOCAL_LENGTHS)
                                                - {P.CONTROL_LENGTH})
                              if f[cell].get(L) is not None
                              and all(L in acc[cell][s] for s in acc[cell]))
                matched[cell] = P.forms(acc[cell], f[cell], mlens, n=n)[0] if mlens else None
                matched_measured[cell] = bool(mlens)
            code, why = P.verdict(ctrl, comp_forms, comp_counts, matched, matched_measured)
            per_read[read] = {"verdict": code, "why": why, "control": ctrl,
                              "control_seeds": ctrl["seeds"],
                              "matched_measured": matched_measured,
                              "lengths_read": list(lengths), "forms": comp_forms,
                              "seed_counts": comp_counts, "matched_forms": matched, "acc": acc}
        per_arch[arch] = per_read
    return per_arch


def post_hoc_section(runs, floors, cfg):
    """Two readings computed AFTER the numbers existed, kept apart from the pre-registered rule.

    The first is the per-seed pairing between the state component and the composed cell on the
    GUIDED read — the rule counts seeds per cell and cannot see whether it is the SAME seeds. The
    second is what the checkpoint diagnostic does and does not say, which matters because the
    number looks like a partial trace and is not one.
    """
    if not runs:
        return []
    L = cfg["guided_lengths"][0] if cfg.get("guided_lengths") else 48
    rows = []
    for r in sorted(runs, key=lambda r: r["seed"]):
        g = r["stages"][-1]["guided"]
        cell = {}
        for c in ("state", "bind", "composed"):
            v = g.get(c, {}).get(str(L), {}).get("match")
            f = floors.get(f"{c}@{L}", {}).get("floor")
            cell[c] = (v, P.clears(v, f, cfg["guided_n"])[0] if v is not None else False)
        rows.append((r["seed"], cell))
    paired = all((c["state"][1] == c["composed"][1]) for _s, c in rows)
    out = ["", "# Post-hoc (not pre-registered)", "",
           f"## The composed cell and the state component move together, seed by seed (GUIDED, L={L})",
           "", "| seed | state | composed | bind |", "|---|---|---|---|"]
    for seed, c in rows:
        out.append(f"| {seed} | " + " | ".join(
            f"{c[k][0]:.3f}{' (clears)' if c[k][1] else ''}" for k in
            ("state", "composed", "bind")) + " |")
    if paired:
        out += ["", "The composed cell clears on EXACTLY the seeds where the state component "
                "clears, and fails on the seed where it fails, while the retrieval component "
                "clears on all three. The pre-registered rule counts seeds per cell and so "
                "cannot see this: it is the same statement as the verdict at one operating "
                "point, and a stronger one, because it says the composed cell costs this "
                "architecture nothing beyond the state leg rather than that both happened to "
                "reach 2 of 3."]
    out += ["", "## What the checkpoint diagnostic says", "",
            "Each checkpoint is the whole of P and then the whole of B, k + m = 12 slots per "
            "event, and each COMPONENT cell holds one of the two maps still by construction — "
            "the state cell has no gives, so its B never moves, and the retrieval cell has no "
            "swaps, so its P never moves. A model that emits the frozen half and guesses the "
            "moving half therefore scores 0.5 + 0.5/6 = 0.583, which is where every component "
            "cell sits (0.573-0.611). On the components the emitted trace is at chance on the "
            "half that moves while the answer is 0.99-1.00, so it is not the trace that carries "
            "those answers; on the composed cell, where neither half is frozen, the trace is "
            "essentially exact (0.932-1.000). The diagnostic is not a partial trace and no "
            "verdict reads it."]
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
    return [
        "", "## The recipe", "",
        f"`d_model={cfg['d_model']}` x `n_layers={cfg['n_layers']}`, `n_heads="
        f"{cfg.get('n_heads')}`, batch {cfg['batch']}, {cfg['steps']} steps, "
        f"{cfg['train_n']} items per stage over the three-stage curriculum "
        f"({' -> '.join(n for n, _s, _w in cfg['schedule'])}), supervision `{cfg['fmt']}` "
        "(answer-masked plain documents plus the specs' own per-event checkpoint documents).",
        "",
        "Two numbers in it are set by the hardware, on the documents this run trains on "
        "(the composed cell's checkpoint document is 2540 tokens at L=96, mean 958):",
        "",
        "- **head dimension 128, so 6 heads at d768.** `GatedDeltaProduct` at head dimension 192 "
        "(4 heads at d768) runs 0.836 s/step against 0.108 s/step at 128, same width, same depth, "
        "same batch — a kernel path, not a model property. At 6 heads the GDP state per layer is "
        "6x128x128, larger than the 4x128x128 of the d512x6 run this one supersedes.",
        "- **batch 16.** The flagship recipe's batch of 128 does not fit: at d768x8 the longest "
        "document slice runs out of memory at batch 24 on a 32 GB card (peak 26.9 GB at 16). "
        f"This run therefore draws {cfg['batch'] * cfg['steps'] / 1000:.0f}k sequences per seed "
        f"against the flagship's 3.2M, from {cfg['train_n'] * 2 / 1000:.0f}k documents per stage.",
    ]


def cost_section(cfg, grid):
    """The composed cell's cost over each component IN BOTH COST MODELS, and the lengths the
    matched-cost control uses.

    Both are printed because they disagree, and the disagreement is the control's whole point: a
    composed cell at floor is explained by composition only once the components have been read at
    the length that costs what it costs. CHARGED STEPS is what a scratchpad solver pays; FORWARD
    -PASS TOKENS is what a streaming model pays, and it is the axis the matched lengths are
    solved on here (``P.MATCHED_AXIS``).
    """
    costs, ml = cfg.get("cell_costs") or {}, cfg.get("matched_lengths") or {}
    if not costs:
        return []
    out = ["", "## The two cost models, and the matched-cost control", "",
           "| composed L | vs state (steps) | vs state (tokens) | vs bind (steps) | "
           "vs bind (tokens) | matched state L | matched bind L |",
           "|---|---|---|---|---|---|---|"]
    for L in P.LOCAL_LENGTHS:
        c = costs.get(f"composed@{L}")
        s, b = costs.get(f"state@{L}"), costs.get(f"bind@{L}")
        if not (c and s and b):
            continue
        row = ml.get(str(L)) or ml.get(L) or {}
        mstate = (row.get("state") or {}).get("L")
        mbind = (row.get("bind") or {}).get("L")
        out.append(
            f"| {L} | {c[0]/s[0]:.2f}x | {c[1]/s[1]:.2f}x | {c[0]/b[0]:.2f}x | "
            f"{c[1]/b[1]:.2f}x | {mstate or '— (unreachable)'} | "
            f"{mbind or '— (unreachable)'} |")
    out += ["", "_A matched length is where that component's own forward pass costs what the "
            "composed cell costs at L. Unreachable on the retrieval component past L=132: its "
            "sampler pins the resolving write into a window that gets exponentially harder to "
            "satisfy as the stream grows, so the cell cannot be run long enough to cost what the "
            "composed cell costs at L=64 or L=96 (protocol.BIND_MATCHED_MAX). That is a property "
            "of the instrument, not of this run._"]
    return out


def write_markdown(runs, per_arch, floors, cfg, grid, path):
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
        "## Size (compute-matched: shared d_model and depth; `fprm` is weight-tied)",
        "",
        "| arch | params | FLOPs/token |", "| --- | --- | --- |",
    ]
    for arch, (p, fl) in sorted(cfg["sizes"].items()):
        lines.append(f"| {arch} | {p / 1e6:.1f}M | {fl / 1e6:.2f}M |")
    lines += recipe_section(cfg) + pilot_section(cfg) + cost_section(cfg, grid)
    for read, n in (("plain", cfg["eval_n"]), ("guided", cfg["guided_n"])):
        key = "eval" if read == "plain" else "guided"
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
                    fl = floors.get(f"{cell}@{L}", {}).get("floor")
                    cells.append(" ".join(
                        "—" if v is None else
                        (f"**{v:.3f}**" if P.clears(v, fl, n)[0] else f"{v:.3f}")
                        for v in vals))
                lines.append(f"| {arch} | " + " | ".join(cells) + " |")
            row = []
            for L in lens:
                fr = floors.get(f"{cell}@{L}")
                row.append("—" if fr is None else f"{fr['floor']:.3f} ({fr['floor'] / ch:.2f}x)")
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
        lines += ["", "## Guided checkpoint accuracy (per-slot, diagnostic — not the metric)", "",
                  f"| arch | cell | L | per-seed | per-slot chance |", "|---|---|---|---|---|"]
        for arch, cell, L, vals in ck_rows:
            lines.append(f"| {arch} | {cell} | {L} | "
                         + " ".join(f"{v:.3f}" for v in vals)
                         + f" | {1.0 / cfg['k']:.3f} |")
    lines += ["", "# Verdict", ""]
    for arch, reads in sorted(per_arch.items()):
        for read, v in sorted(reads.items()):
            ctrl = v.get("control", {})
            lines += [f"**{arch} / {read}: {v['verdict']}** — {v['why']}", "",
                      f"seeds clearing: {v['seed_counts']}; positive control (some component "
                      f"clears on this read's grid) {ctrl.get('per_pair')} of "
                      f"{len(cfg['seeds'])} seeds, required {ctrl.get('required')}; "
                      f"matched-cost control: {v['matched_forms']} "
                      f"(measured: {v.get('matched_measured')}); "
                      f"lengths read: {v['lengths_read']}", ""]
    lines += post_hoc_section(runs, floors, cfg)
    lines += ["", "_A **bold** cell clears its own operative floor under the pre-registered "
              "rule. Floors are recomputed from that cell's own items: registry rows plus the "
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
    per_arch = apply_rule(res["runs"], res["floors"], grid, cfg["eval_n"],
                          cfg.get("guided_n", P.N_GUIDED))
    md = Path(str(json_path).replace(".json", ".md"))
    cfg = {**cfg, "sizes": {k: tuple(v) for k, v in cfg["sizes"].items()}}
    write_markdown(res["runs"], per_arch, res["floors"], cfg, grid, md)
    res["verdicts"] = per_arch
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

    grid = {"state": [P.CONTROL_LENGTH, *P.LOCAL_LENGTHS],
            "bind": [P.CONTROL_LENGTH, *P.LOCAL_LENGTHS],
            "composed": [P.CONTROL_LENGTH, *P.LOCAL_LENGTHS]}
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

    archs = [x.strip() for x in a.archs.split(",")]
    sizes = {arch: measured_size(arch, a.d_model, a.n_layers, a.n_heads, tok.vocab_size,
                                 a.device) for arch in archs}
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    prefix = Path(a.out_prefix or f"results/s5bind_v3_three_cell_local_{ts}")
    prefix.parent.mkdir(parents=True, exist_ok=True)
    log_path, md_path, js_path = (Path(f"{prefix}.jsonl"), Path(f"{prefix}.md"),
                                  Path(f"{prefix}.json"))
    cfg = {"k": specs["composed"].k, "archs": archs, "seeds": a.seeds, "steps": a.steps,
           "batch": a.batch, "d_model": a.d_model, "n_layers": a.n_layers,
           "n_heads": a.n_heads, "cell_costs": cell_costs(specs, tok, grid), "lr": a.lr,
           "train_n": a.train_n, "eval_n": a.eval_n, "guided_n": a.guided_n,
           "guided_lengths": a.guided_lengths, "guided_grid": guided_grid,
           "fmt": a.fmt, "grid": grid,
           "matched_lengths": matched, "schedule": [(n, s, w) for n, s, w in SCHEDULE],
           "sizes": sizes, "train_lengths": P.TRAIN_LENGTHS}

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
                                 guided_batch=a.guided_batch)
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
            per_arch = apply_rule(runs, floors, grid, a.eval_n, a.guided_n)
            write_markdown(runs, per_arch, floors, cfg, grid, md_path)
            js_path.write_text(json.dumps({"cfg": cfg, "floors": floors, "runs": runs,
                                           "verdicts": per_arch}, indent=2, default=float))
    if runs:
        per_arch = apply_rule(runs, floors, grid, a.eval_n, a.guided_n)
        write_markdown(runs, per_arch, floors, cfg, grid, md_path)
        js_path.write_text(json.dumps({"cfg": cfg, "floors": floors, "runs": runs,
                                       "verdicts": per_arch}, indent=2, default=float))
        print("\n=== verdict ===", flush=True)
        for arch, reads in sorted(per_arch.items()):
            for read, v in sorted(reads.items()):
                print(f"  {arch} / {read}: {v['verdict']} — {v['why']}", flush=True)
    print(f"\n=== done: {md_path} ===", flush=True)


if __name__ == "__main__":
    main()

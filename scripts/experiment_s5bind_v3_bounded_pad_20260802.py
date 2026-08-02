"""THE BOUNDED-PAD PROTOCOL — a guided format narrow enough that the composed floor survives it.

WHAT IT IS FOR. The shipped GUIDED format writes the whole of P then the whole of B after every
event, so the k + m live slots the one-structure bound prices are handed to every policy and the
composed cell is UNFLOORABLE under it (``factworld.validity``, "THE GUIDED PROTOCOL", T2). The
PLAIN protocol keeps the floor and floors everything, because the state component is itself at
floor on the plain channel for every architecture at every length. Neither protocol has the
composed cell both floored and reachable. This module builds the third one.

THE WIDTH IS DERIVED, NOT CHOSEN (``scripts/probe_s5bind_v3_bounded_pad_floor_20260802.py``). A
pad of w slots is w free live slots, so a row of true cost W costs W - w of a policy's own and the
class rule admits it iff W - w <= max(k, m) + 1. The composed cell's algorithm costs k + m + 1, so
the class excludes the task iff w <= min(k, m) - 1 — w <= 5 at the k = m = 6 local operating
point. The same inequality admits ``partial_carry_j`` for exactly j <= w, and that family is not
flat, so the floor's VALUE decides which floorable widths are worth training. Measured on the
exact scored items and a disjoint 4000-item pool at k = 6:

    w      composed@48         composed@64         composed@96
    1-2    0.2344 (1.17x)      0.2266 (1.13x)      0.2109 (1.05x)   = the PLAIN floor, unchanged
    3      0.2891 (1.45x)      0.2578 (1.29x)      0.2109 (1.05x)
    4      0.3906 (1.95x)      0.2734 (1.37x)      0.2109 (1.05x)
    5      0.6132 (3.07x)      0.4609 (2.31x)      0.2969 (1.48x)   bar 0.763 — unbuyable
    6+     unfloorable         unfloorable         unfloorable

    w <= 2 is the widest pad that costs the floor NOTHING. This module registers w = 2.

TWO FAMILIES ARE KILLED BY THAT INEQUALITY BEFORE ANY MODEL IS TRAINED, and both are the obvious
first ideas. A SLIDING WINDOW OF THE LAST w CHECKPOINTS is unfloorable at every w, because one
checkpoint is already k + m slots wide. EMITTING ONLY THE HALF OF THE STATE THE QUERY DOES NOT NAME
is min(k, m) = 6 slots, which TIES the bound and admits the task; the boundary is a tie and not a
strict inequality, so "narrower than one structure" is off by one and 6 is not floorable. What
survives is a bounded per-event summary, and the width ladder above is its whole range.

THE REGISTERED FORMATS, all of them a bounded summary at a derived width:
    moved2   the post-event values of the slots the event MOVED. A swap of (a, x) writes
             ``P[a] P[x]``; a give to o writes ``B[o]`` and, so the width is constant, the value it
             displaced. Both tokens are live reads of the structure being written.
    delta2   the RESOLVED OPERAND and the displaced value. Token one is a live read of the SOURCE
             structure — the composed cell's own defining work — and token two of the target. On a
             COMPONENT cell every operand is named, so that token is a copy of one the event
             already printed and only one of its two slots carries supervision.
    hybrid4  ``moved2`` plus two state slots by rotation, which bounds how far back the queried
             slot was last written to min(k, m) events. Pad 4, and the floor survives it.

WHAT IS LOST, and it is not small. The TRACE read does not exist under a bounded format. It reads
the model's own final checkpoint's value for the QUERIED slot, and a bounded pad has no slot for a
particular query — that read was available only because the format was wide enough to void the
floor. The state probe that replaces it is ``slot_acc``, the per-token accuracy of the emitted pad
against the gold pad. It is a sharper probe than the dense format's: a never-update policy scores
0.804-0.901 per slot on the dense checkpoint but only 0.188-0.323 on the bounded pad.

THE BOUNDED PAD DOES NOT STOP STATE TRACKING FORMING. It forms it more completely than the dense
format does. Measured on gdp_hybrid at the registered curriculum, k = 6, guided n = 128, as
``answer / pad`` per seed (the floor row is the bounded-pad floor at pad 2):

    docs          seed  state@17      state@80      bind@31       bind@132      composed@48
    pad only      0     0.195 / 1.000 0.172 / 1.000 0.164 / 0.679 0.188 / 0.681 0.188 / 0.258
                  1     0.172 / 1.000 0.148 / 0.999 0.164 / 1.000 0.266 / 1.000 0.211 / 0.482
    + answer      0     1.000 / 1.000 1.000 / 1.000 1.000 / 1.000 1.000 / 1.000 0.258 / 0.702
                  1     0.172 / 1.000 0.148 / 1.000 0.156 / 1.000 0.180 / 1.000 0.219 / 0.538
                  2     0.156 / 1.000 0.188 / 0.999 0.180 / 1.000 0.133 / 1.000 0.133 / 0.852
    floor               0.2188        0.2500        0.2000        0.2000        0.2344

A ``pad`` of 1.000 means the generated context is BYTE-IDENTICAL to the gold pad, so both
structures are written down correctly at every event of an 80- and a 132-event stream, on every
seed of both runs. The dense format's own checkpoint accuracy is 0.575-0.592 — BELOW its 0.804
never-update reference — so what the dense format supplied was never the tracking. It supplied the
READOUT: its final checkpoint's queried slot IS the gold answer on 1.000 of items against
0.281-0.625 for a bounded pad, so the dense guided answer read is a copy of a token the format
printed beside the query, and removing that copy is what the width costs.

THE READOUT IS THE SEPARATE, BIMODAL EVENT, and ``--pad_answer_docs`` is what buys it: masking a
second copy of each pad document to the answer takes that token from one in ~250 of a full
next-token loss to half the document's loss mass, and it is the same argument the runner already
makes for PLAIN documents. It carries the components to 1.000 on 1 of 3 seeds and leaves them at
floor on 2, so FORMS (``SEEDS_CLEAR`` = 2) is not met and the arm's verdict is
V4_COMPONENT_UNREADABLE. ``attribute_answers`` says the floored seeds are not falling back to
anything: the answer is not the stated initial value (0.000-0.008) and not the last pad token
(0.148-0.266), it is near informed chance.

ON THE SEED THAT DOES FORM, the pattern is the one the instrument was built to produce, and for
the first time on a protocol under which the composed cell HAS a floor: both components clear at
their work-matched lengths AND at their token-matched ones (state@80 costs 691 tokens against
composed@48's 717; bind@132 costs 711), z = 19.6-22.6, while the composed cell is at floor at all
three of its registered lengths — 0.258 / 0.195 / 0.266 against 0.2344 / 0.2266 / 0.2109, z = 0.63
/ -0.84 / 1.52. It is one seed and is reported as one.

    grid:   .venv-train/bin/python scripts/experiment_s5bind_v3_bounded_pad_20260802.py --grid \\
                --archs gdp_hybrid --seeds 0 1 2 --steps 25000 --format moved2 --pad_answer_docs
    read:   ... --report results/<run>.json
    depth:  ... --decode_from results/<run>_ckpt --decode_cells composed@48,composed@64,composed@96
    why:    ... --attribute results/<run>_ckpt --archs gdp_hybrid --seeds 0

    A guided decode is batch-shaped: the padded batched argmax runs under bf16 autocast, so a cell
    moves by about one item in 128 between ``--guided_batch`` 32 and 128 (composed@48 seed 0 reads
    0.250 and 0.258). Decode at the batch the run used when reproducing one of its numbers.
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
from factworld.composition import SWAP, read as _read                      # noqa: E402
from factworld.render import Renderer                                      # noqa: E402
from factworld.runner import evaluate_task                                 # noqa: E402
from factworld.tokenizer import Tokenizer                                  # noqa: E402
import protocol_s5bind_v3_three_cell_20260731 as P                         # noqa: E402
import experiment_s5bind_v3_three_cell_local_20260731 as E                 # noqa: E402

# The registered width, and the width of every format name below. ``dense`` is the shipped
# format and is k + m; it is here as the CONTROL and is unfloorable by construction.
PAD_WIDTH = {"dense": None, "moved2": 2, "delta2": 2, "hybrid4": 4}
FORMATS = tuple(PAD_WIDTH)
MAX_DOC_TOKENS = E.MAX_DOC_TOKENS

# WHY ACCUMULATION DOES NOT WIDEN THE PAD, since a narrow format writes many blocks over a stream
# and their union covers every slot. Two conventions decide it and both are the repo's own:
#   W5  the event stream is NOT addressable. The pad is interleaved with the events, so recovering
#       an earlier block's value costs a backward SCAN (one E + one C per event passed) and the
#       policy holds nothing while scanning. Only the block ADJACENT to the answer is readable at
#       O(1), and that block is ``pad`` slots wide. This is the same accounting that makes the
#       DENSE format's pad k + m: there the adjacent block is the whole final checkpoint.
#   The pad grants REGISTERS, not CORRECTNESS. A row is scored by running its own policy, pad
#       included, so a cheap policy's accumulated pad is exactly as cheap as its answer. This is
#       why "read the pad token beside the last event naming the queried agent" is not a floor
#       row: writing that token is the tracking, and a policy that cannot track writes the wrong
#       one. What the dense format hands out is the ability to RUN the task's own algorithm at
#       W = 1 of a policy's own; at pad 2 the best a policy can run is ``partial_carry_j2``.


# ---- the format -----------------------------------------------------------------------------
def event_insertion_points(ex):
    """Plain-token indices after which the shipped format inserts a checkpoint, one per event.

    Recovered from the alignment the dense format already ships (``sweep._interleaved_slots``)
    rather than re-parsed, so a narrow format sits at exactly the positions the dense one does and
    the two protocols differ in WIDTH and in nothing else.
    """
    from sweep import _interleaved_slots

    inter = ex.meta.get("interleaved_prompt")
    if inter is None:
        return None
    toks, slots = _interleaved_slots(ex.prompt, inter)
    slotset = set(slots)
    at, consumed = [], 0
    for j, _t in enumerate(toks):
        if j in slotset:
            at.append(consumed)
        else:
            consumed += 1
    # slots arrive in consecutive runs of (k + m) at one plain index each; keep one per run
    return sorted(set(at))


def pad_values(ex, fmt, agents=None, objs=None):
    """The gold pad for one item: one bounded block per event, ``PAD_WIDTH[fmt]`` tokens wide.

    Replays the item exactly as ``composition.replay`` does — off the prompt, sharing no code with
    the sampler — and records the tokens the format asks for at each event.

    ``moved2``  the post-event values of the slots the event MOVED (a swap moves two pointer
                cells; a give moves one holder cell, and the value it displaced fills the block so
                the width is constant and query-blind).
    ``delta2``  the RESOLVED OPERAND and the displaced value: token one is a live read of the
                SOURCE structure, which is the composed cell's own defining work.
    ``hybrid4`` ``moved2`` plus two STATE slots by rotation, so a bounded amount of the state is
                handed back each event. The rotation is by event ordinal and never by the query,
                so no format decision depends on what is asked. It is the wider rung the floor
                still survives (pad 4: 0.3906 / 0.2734 / 0.2109 at L = 48 / 64 / 96).
    """
    rec = _read(ex.prompt)
    if rec is None:
        return None
    Pm, Bm = dict(rec["P0"]), dict(rec["B0"])
    # A COMPONENT cell states only the structure it moves (``tasks._ex_s5_bind_v3``: a state cell
    # carries no holder facts), so the rotation runs over the slots this cell actually has. It is
    # still query-blind — what it depends on is the arm, which is fixed before any item is drawn.
    order = ([a for a in (agents or []) if a in Pm]
             + [o for o in (objs or []) if o in Bm])
    out = []
    for i, (kind, tgt, ref, src) in enumerate(rec["events"]):
        x = ref if src == "N" else (Pm.get(ref) if src == "P" else Bm.get(ref))
        if x is None:
            return None
        if kind == SWAP:
            if tgt not in Pm or x not in Pm:
                return None
            disp = Pm[tgt]                       # the value the write displaces
            Pm[tgt], Pm[x] = Pm[x], Pm[tgt]
            moved = [Pm[tgt], Pm[x]]             # the post-event values of the moved slots
        else:
            disp = Bm.get(tgt)
            if disp is None:
                return None
            Bm[tgt] = x
            moved = [Bm[tgt], disp]
        if fmt == "delta2":
            out.append([x, disp])
            continue
        block = list(moved)
        if fmt == "hybrid4":
            if not order:
                return None
            for d in (0, 1):
                sl = order[(2 * i + d) % len(order)]
                v = Pm.get(sl) if sl in Pm else Bm.get(sl)
                if v is None:
                    return None
                block.append(v)
        out.append(block)
    return out


def narrow_interleaved(ex, fmt, agents=None, objs=None):
    """``(tokens, slot_indices, gold_slot_values)`` for one item under a bounded format.

    ``dense`` returns the shipped interleaved prompt unchanged, so the two protocols run through
    exactly the same decode and the comparison is the format alone.
    """
    from sweep import _interleaved_slots

    if fmt == "dense":
        inter = ex.meta.get("interleaved_prompt")
        if inter is None:
            return None
        toks, slots = _interleaved_slots(ex.prompt, inter)
        return toks, slots, [toks[s] for s in slots]
    at = event_insertion_points(ex)
    vals = pad_values(ex, fmt, agents, objs)
    if at is None or vals is None or len(at) != len(vals):
        return None
    plain = ex.prompt.split()
    toks, slots, gold = [], [], []
    prev = 0
    for p, block in zip(at, vals):
        toks += plain[prev:p]
        for t in block:
            slots.append(len(toks))
            gold.append(t)
            toks.append(t)
        prev = p
    toks += plain[prev:]
    return toks, slots, gold


def slot_order(spec):
    """The canonical slot order a checkpoint runs in: P in agent order then B in object order.

    The same order ``tasks._ex_s5_bind_v3`` builds ``meta["trace"]`` in, so a rotating format
    indexes the state the dense one prints and the two are comparable slot for slot.
    """
    world, _r = TK.build_world(spec)
    return list(world.agents[:spec.k]), list(world.objects[:spec.n_objects_active])


def narrow_document(ex, fmt, agents=None, objs=None):
    """The training document for one item: the bounded-pad prompt followed by the answer."""
    got = narrow_interleaved(ex, fmt, agents, objs)
    return None if got is None else " ".join(got[0]) + " " + ex.answer


def stage_documents(specs, weights, train_n, tok, fmt, mix=True, answer_docs=False):
    """``(encoded docs, prompt_lens)`` for one stage under a bounded format.

    Same discipline as the shipped runner: a pad document takes a full next-token loss (the pad IS
    the supervision, ``prompt_len`` 1) and a plain document takes the answer-masked loss, because
    otherwise the answer is one token in several hundred.

    ``answer_docs`` adds a THIRD document per item: the pad prompt with the loss masked to the
    ANSWER. It exists because the bounded pad separates two things the dense format had welded
    together. Under the dense format the final checkpoint's queried slot IS the gold answer on
    every item, so the answer token is a copy of a token ten positions back and needs no gradient
    of its own; under a bounded pad the answer is in the last block on only 0.28-0.63 of items, so
    reading it out is a distinct operation — and under a full next-token loss it carries one token
    in ~250 against the pad's 2L. The flag is set on a STATE-COMPONENT run and never on the
    composed cell, which is the same rule the masked loss and the checkpoint mix were chosen under.
    """
    key = (tuple(sorted(weights.items())), train_n, fmt, bool(mix), bool(answer_docs),
           tuple(sorted(specs)))
    if key in _DOC_CACHE:
        return _DOC_CACHE[key]
    pairs = []
    for arm, share in sorted(weights.items()):
        n = int(round(train_n * share))
        if n <= 0:
            continue
        ags, obs = slot_order(specs[arm])
        for e in TK.generate(specs[arm], "train", n=n):
            if mix:
                pairs.append((f"{e.prompt} {e.answer}", len(tok.encode(e.prompt))))
            doc = narrow_document(e, fmt, ags, obs)
            if doc is not None:
                pairs.append((doc, 1))
                if answer_docs:
                    pairs.append((doc, len(tok.encode(doc[:-len(e.answer) - 1]))))
    enc = [(tok.encode(t, add_eos=True)[:MAX_DOC_TOKENS], pl) for t, pl in pairs]
    enc.sort(key=lambda x: len(x[0]))
    _DOC_CACHE[key] = ([a for a, _ in enc], [b for _, b in enc])
    return _DOC_CACHE[key]


_DOC_CACHE: dict = {}


# ---- the read -------------------------------------------------------------------------------
def bounded_free_run_batched(model, tok, spec, length, n, device, fmt, batch=128, max_answer=4):
    """The GUIDED read under a BOUNDED pad: events teacher-forced, every pad token and the answer
    generated, and nothing but the pad accumulating into the answer's context.

    Structurally ``E.guided_free_run_batched`` with the slot stream narrowed. Returns
    ``(answer_match, slot_acc, None)``: the third slot is where the shipped runner returns the
    TRACE read, and it is None here because a bounded pad has no slot for a particular query. That
    read was available only under a format wide enough to void the composed cell's floor, so its
    absence is a property of the protocol and not a missing measurement. ``slot_acc`` is the state
    probe that replaces it — a model emitting the right resolved operands and displaced values at
    every event is tracking both structures.
    """
    import torch

    examples = TK.generate(spec, "test", n=n, length=length)
    ags, obs = slot_order(spec)
    prepped = []
    for ex in examples:
        got = narrow_interleaved(ex, fmt, ags, obs)
        if got is None:
            return None, None, None
        toks, slots, gold = got
        prepped.append((toks, slots, set(slots), gold, ex.answer))
    n_slots = len(prepped[0][1])
    if any(len(p[1]) != n_slots for p in prepped):
        return None, None, None
    hits = ck_hits = ck_total = 0
    model.eval()
    with torch.no_grad():
        for b0 in range(0, len(prepped), batch):
            chunk = prepped[b0:b0 + batch]
            ids = [[] for _ in chunk]
            cursor = [0] * len(chunk)
            gen = [[] for _ in chunk]
            for ordinal in range(n_slots + 1):
                for i, (toks, slots, slotset, _g, _a) in enumerate(chunk):
                    limit = slots[ordinal] if ordinal < n_slots else len(toks)
                    while cursor[i] < limit:
                        if cursor[i] not in slotset:
                            ids[i] += tok.encode(toks[cursor[i]])
                        cursor[i] += 1
                if ordinal < n_slots:
                    nxt = E._batched_argmax(model, ids, tok, device)
                    for i, tid in enumerate(nxt):
                        ids[i].append(tid)
                        gen[i].append(tok.id_to_token.get(tid, "<unk>"))
                        cursor[i] += 1
            outs = [[] for _ in chunk]
            live = list(range(len(chunk)))
            for _ in range(max_answer):
                if not live:
                    break
                nxt = E._batched_argmax(model, [ids[i] for i in live], tok, device)
                still = []
                for i, tid in zip(live, nxt):
                    if tid == tok.eos_id:
                        continue
                    ids[i].append(tid)
                    outs[i].append(tid)
                    still.append(i)
                live = still
            for i, (_t, _s, _ss, gold, ans) in enumerate(chunk):
                hits += bool(TK.score_relaxed(Renderer.normalize(tok.decode(outs[i])),
                                              Renderer.normalize(ans)))
                ck_hits += sum(1 for a, g in zip(gen[i], gold) if a == g)
                ck_total += len(gold)
    model.train()
    return hits / max(1, len(prepped)), ck_hits / max(1, ck_total), None


def attribute_answers(ckpt_path, spec, length, n, device, fmt, batch=64):
    """WHAT the model answers when it answers wrong, under a bounded pad.

    It exists because ``slot_acc`` came back at exactly 1.0 on the state component while the
    answer sat at floor: the generated context is then byte-identical to the gold pad, so the
    state is written down correctly at every event and the failure is entirely in reading it back.
    This says which wrong thing is read instead, against four references computed from the item:

        gold          the answer (the floor comparison's own quantity);
        stated        the STATED initial value of the queried slot — the ``initial_only`` row;
        last_pad      the final pad token, i.e. the nearest thing in context;
        in_last_pad   the answer appears anywhere in the final pad block (the copy channel the
                      DENSE format supplies on 1.000 of items and this one on 0.28-0.63);
        pad_for_q     the value the pad last wrote for the QUERIED slot — what a correct readout
                      of this model's own (here perfect) scratchpad would return.

    A high ``pad_for_q`` with a low ``gold`` would mean the readout is right and the pad is stale;
    they are the same number whenever the pad is perfect, which is the case worth separating.
    """
    import torch

    model, _blob = E.load_checkpoint(ckpt_path, device)
    tokz = _blob
    del tokz
    from factworld.tokenizer import Tokenizer as _T
    world, r = TK.build_world(spec)
    tok = _T.build([world], r)
    ags, obs = slot_order(spec)
    examples = TK.generate(spec, "test", n=n, length=length)
    rows = []
    for ex in examples:
        got = narrow_interleaved(ex, fmt, ags, obs)
        rec = _read(ex.prompt)
        if got is None or rec is None:
            return None
        toks, slots, gold = got
        w = PAD_WIDTH[fmt]
        qkind, qtgt = rec["query"]
        # the value the pad last wrote for the queried slot, replaying which slots each block holds
        Pm, Bm = dict(rec["P0"]), dict(rec["B0"])
        last_for_q = (Pm if qkind == "state" else Bm).get(qtgt)
        for i, (kind, tgt, ref, src) in enumerate(rec["events"]):
            x = ref if src == "N" else (Pm.get(ref) if src == "P" else Bm.get(ref))
            if kind == SWAP:
                Pm[tgt], Pm[x] = Pm[x], Pm[tgt]
                if qkind == "state" and qtgt in (tgt, x):
                    last_for_q = Pm[qtgt]
            else:
                Bm[tgt] = x
                if qkind == "bind" and qtgt == tgt:
                    last_for_q = Bm[tgt]
        rows.append((toks, slots, set(slots), gold, ex.answer, rec, w, last_for_q))
    n_slots = len(rows[0][1])
    hits = {"gold": 0, "stated": 0, "last_pad": 0, "in_last_pad": 0, "pad_for_q": 0, "n": 0}
    model.eval()
    with torch.no_grad():
        for b0 in range(0, len(rows), batch):
            chunk = rows[b0:b0 + batch]
            ids = [[] for _ in chunk]
            cursor = [0] * len(chunk)
            for ordinal in range(n_slots + 1):
                for i, (toks, slots, slotset, *_r) in enumerate(chunk):
                    limit = slots[ordinal] if ordinal < n_slots else len(toks)
                    while cursor[i] < limit:
                        if cursor[i] not in slotset:
                            ids[i] += tok.encode(toks[cursor[i]])
                        cursor[i] += 1
                if ordinal < n_slots:
                    for i, tid in enumerate(E._batched_argmax(model, ids, tok, device)):
                        ids[i].append(tid)
                        cursor[i] += 1
            outs = [[] for _ in chunk]
            live = list(range(len(chunk)))
            for _ in range(4):
                if not live:
                    break
                nxt = E._batched_argmax(model, [ids[i] for i in live], tok, device)
                still = []
                for i, tid in zip(live, nxt):
                    if tid == tok.eos_id:
                        continue
                    ids[i].append(tid)
                    outs[i].append(tid)
                    still.append(i)
                live = still
            for i, (_t, _s, _ss, gold, ans, rec, w, lq) in enumerate(chunk):
                pred = Renderer.normalize(tok.decode(outs[i]))
                qkind, qtgt = rec["query"]
                stated = (rec["P0"] if qkind == "state" else rec["B0"]).get(qtgt)
                hits["n"] += 1
                hits["gold"] += bool(TK.score_relaxed(pred, Renderer.normalize(ans)))
                if stated:
                    hits["stated"] += bool(TK.score_relaxed(
                        pred, Renderer.normalize(f"{stated}.")))
                hits["last_pad"] += bool(TK.score_relaxed(
                    pred, Renderer.normalize(f"{gold[-1]}.")))
                hits["in_last_pad"] += any(
                    Renderer.normalize(f"{t}.") == Renderer.normalize(ans) for t in gold[-w:])
                if lq:
                    hits["pad_for_q"] += bool(TK.score_relaxed(
                        pred, Renderer.normalize(f"{lq}.")))
    del model
    torch.cuda.empty_cache()
    return {k: (v / hits["n"] if k != "n" else v) for k, v in hits.items()}


def decode_cells(ckpt_dir, archs, seeds, cells, guided_n, device, fmt, batch=64):
    """Score saved weights on extra GUIDED cells, with no training.

    The shipped guided read buys ONE composed length because its decode is (k + m) L sequential
    rounds per item, i.e. O(n L^2). A bounded pad makes it ``pad`` L rounds — a sixth of that at
    pad 2 — so the composed cell's whole registered grid becomes affordable, and "does not clear at
    ANY registered length" becomes a measurement rather than an extrapolation from one rung.
    """
    from factworld.tokenizer import Tokenizer as _T
    from factworld import validity as V

    specs = E.three_cell_specs(P.TRAIN_LENGTHS)
    world, r = TK.build_world(specs["composed"])
    tok = _T.build([world], r)
    out = {}
    for cell, L in cells:
        spec = specs[cell]
        k, m = spec.k, spec.n_objects_active
        pool = TK.generate(spec, "test", n=guided_n + P.N_SCORE, length=L)
        scored, big = pool[:guided_n], pool[guided_n:]
        named, query = V.s5_bind_v3_is_named(big), V.s5_bind_v3_query_kind(big)
        vals = []
        for items in (scored, big):
            ns, ng = V.s5_bind_v3_shape(items)
            vals.append(V.s5_bind_v3_pad_operative_floor(
                V.s5_bind_v3_pad_floors(items, k, m, named, query), k, m, ns, ng, named, query,
                pad=PAD_WIDTH[fmt]))
        floor = None if all(v is None for v in vals) else max(v for v in vals if v is not None)
        out[f"{cell}@{L}"] = {"floor": floor, "runs": {}}
        for arch in archs:
            for seed in seeds:
                pth = E.checkpoint_path(ckpt_dir, arch, seed)
                if not Path(pth).exists():
                    continue
                model, _b = E.load_checkpoint(pth, device)
                a, sa, _t = bounded_free_run_batched(model, tok, spec, L, guided_n, device, fmt,
                                                     batch=batch)
                del model
                import torch
                torch.cuda.empty_cache()
                cl, z = P.clears(a, floor, guided_n)
                out[f"{cell}@{L}"]["runs"][f"{arch}_s{seed}"] = {
                    "match": a, "slot_acc": sa, "clears": cl, "z": z}
                print(f"  {arch} s{seed} {cell}@{L}: match={a:.3f} slot={sa:.3f} "
                      f"floor={floor:.4f} z={z:.2f} {'CLEARS' if cl else 'at floor'}", flush=True)
    return out


def evaluate_all(model, arch, specs, tok, world, grid, *, eval_n, guided_n, guided_lengths,
                 device, fmt, guided_batch=128):
    """``(plain, guided)`` for ONE model over a grid, with the guided read taken under ``fmt``."""
    backend = LocalBackend([world], arch=arch, model=model, tokenizer=tok, device=device)
    ev = E.eval_cells(backend, specs, eval_n, grid)
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
                a, ck, _tr = bounded_free_run_batched(model, tok, specs[cell], L, guided_n,
                                                      device, fmt, batch=guided_batch)
                gv[cell][str(L)] = {"match": a, "slot_acc": ck, "trace": None, "pad_format": fmt}
                shown = ("unevaluable" if a is None else f"match={a:.3f} slot={ck:.3f}")
                print(f"     guided[{fmt}] {cell:9s} L{L}: {shown} [{time.time() - t1:.0f}s]",
                      flush=True)
    return ev, gv


# ---- the pilot ------------------------------------------------------------------------------
def pilot_one(arch, seed, fmt, spec, tok, world, *, steps, batch, d_model, n_layers, n_heads, lr,
              train_n, eval_n, guided_n, guided_lengths, device, guided_batch, loss_log_interval,
              answer_docs=False):
    """One (arch, seed, format) pilot on the STATE COMPONENT ALONE — one stage, one cell.

    The composed cell is not in the mix and is not scored. The question the pilot answers is the
    one that decides whether the bounded protocol is worth a grid at all: does the state component
    still FORM when the format stops handing the model its state back?
    """
    import torch
    from factworld import train as T

    specs = {"state": spec}
    docs, plens = stage_documents(specs, {"state": 1.0}, train_n, tok, fmt,
                                  answer_docs=answer_docs)
    t0 = time.time()
    run = T.run(arch, tok, docs, [], steps=steps, batch=batch, d_model=d_model,
                n_layers=n_layers, n_heads=n_heads, d_ff=4 * d_model, lr=lr, seed=seed,
                return_model=True, device=device, model=None, use_short_conv=True,
                loss_log_interval=loss_log_interval, prompt_lens=plens)
    model = run["model"]
    print(f"  -- {arch} s{seed} [{fmt}]: {steps} steps, {len(docs)} docs, "
          f"loss={run['final_loss']:.4f} [{time.time() - t0:.0f}s]", flush=True)
    ev, gv = evaluate_all(model, arch, specs, tok, world, {"state": guided_lengths},
                          eval_n=eval_n, guided_n=guided_n,
                          guided_lengths={"state": guided_lengths}, device=device, fmt=fmt,
                          guided_batch=guided_batch)
    del model
    torch.cuda.empty_cache()
    return {"arch": arch, "seed": seed, "format": fmt, "steps": steps, "n_docs": len(docs),
            "final_loss": run["final_loss"], "train_s": round(time.time() - t0),
            "plain": ev, "guided": gv,
            "loss_curve": [(int(s), float(v)) for s, v in run.get("loss_curve", [])]}


# ---- the grid -------------------------------------------------------------------------------
def bounded_floors_for(guided_grid, guided_n, pad):
    """The floor at each cell of the BOUNDED-PAD grid, under the pad rule and at the read's own n.

    ``validity.s5_bind_v3_pad_operative_floor`` is the rule and is not reimplemented, so a floor
    that moves because the pad width moved is visible as exactly that. Both item sets are measured
    with the operative one the larger, for the reason ``P.cell_floor`` gives: the max over admitted
    rows carries an upward selection bias that a 128-item read does not average out.
    """
    from factworld import validity as V

    out = {}
    for cell, lengths in guided_grid.items():
        spec = TK.CANONICAL[P.LOCAL_CELLS[cell]]
        k, m = spec.k, spec.n_objects_active
        for L in lengths:
            pool = TK.generate(spec, "test", n=guided_n + P.N_SCORE, length=L)
            scored, big = pool[:guided_n], pool[guided_n:]
            named = V.s5_bind_v3_is_named(big)
            query = V.s5_bind_v3_query_kind(big)
            vals = []
            for items in (scored, big):
                ns, ng = V.s5_bind_v3_shape(items)
                fl = V.s5_bind_v3_pad_floors(items, k, m, named, query)
                vals.append(V.s5_bind_v3_pad_operative_floor(fl, k, m, ns, ng, named, query,
                                                             pad=pad))
            floor = None if all(v is None for v in vals) else max(v for v in vals if v is not None)
            out[f"{cell}@{L}"] = {
                "cell": spec.name, "L": L, "k": k, "m": m, "protocol": f"bounded_pad{pad}",
                "pad": pad, "floor": floor, "chance": 1.0 / (k - 1),
                "floorable": V.s5_bind_v3_pad_floorable(k, m, pad, named),
                "pad_max_width": V.s5_bind_v3_pad_max_width(k, m),
                "floor_on_scored_items": vals[0], "floor_disjoint": vals[1],
                "n_swap": V.s5_bind_v3_shape(scored)[0],
                "n_give": V.s5_bind_v3_shape(scored)[1]}
            print(f"  bounded floor (pad {pad}) {cell}@{L} = "
                  + ("unfloorable" if floor is None else f"{floor:.4f}")
                  + f" (n={guided_n})", flush=True)
    return out


def grid_run_one(arch, seed, specs, tok, world, grid, *, steps, batch, d_model, n_layers,
                 n_heads, lr, train_n, eval_n, guided_n, guided_lengths, device, fmt,
                 loss_log_interval, guided_batch=128, ckpt_dir=None, answer_docs=False):
    """One (arch, seed) carried through the registered three-stage curriculum under a BOUNDED pad.

    Structurally ``E.run_one`` with ``E.stage_documents`` replaced by the bounded one and the
    guided read taken under ``fmt``; the schedule, the stage shares and the checkpoint discipline
    are the shipped ones, so the only thing that differs from the run at HEAD is the pad width.
    """
    import torch
    from factworld import train as T

    model, stages = None, []
    for si, (name, share, weights) in enumerate(E.SCHEDULE):
        last = si == len(E.SCHEDULE) - 1
        stage_steps = max(1, int(round(steps * share)))
        docs, plens = stage_documents(specs, weights, train_n, tok, fmt,
                                      answer_docs=answer_docs)
        t0 = time.time()
        run = T.run(arch, tok, docs, [], steps=stage_steps, batch=batch, d_model=d_model,
                    n_layers=n_layers, n_heads=n_heads, d_ff=4 * d_model, lr=lr, seed=seed,
                    return_model=True, device=device, model=model, use_short_conv=True,
                    loss_log_interval=loss_log_interval, prompt_lens=plens)
        model = run["model"]
        print(f"  -- {name}: {stage_steps} steps, {len(docs)} docs, "
              f"loss={run['final_loss']:.4f} [{time.time() - t0:.0f}s]", flush=True)
        if ckpt_dir:
            E.save_checkpoint(model, E.checkpoint_path(ckpt_dir, arch, seed), arch=arch,
                              seed=seed, stage=name,
                              build={"d_model": d_model, "n_layers": n_layers,
                                     "n_heads": n_heads, "d_ff": 4 * d_model,
                                     "use_short_conv": True, "vocab_size": tok.vocab_size},
                              provenance={"steps": stage_steps, "n_docs": len(docs),
                                          "mix": weights, "final_loss": run["final_loss"],
                                          "lr": lr, "batch": batch, "fmt": fmt,
                                          "pad": PAD_WIDTH[fmt], "train_n": train_n,
                                          "train_lengths": list(P.TRAIN_LENGTHS)})
        if last:
            ev, gv = evaluate_all(model, arch, specs, tok, world, grid, eval_n=eval_n,
                                  guided_n=guided_n, guided_lengths=guided_lengths,
                                  device=device, fmt=fmt, guided_batch=guided_batch)
        else:
            ev, gv = evaluate_all(
                model, arch, specs, tok, world,
                {c: [P.CONTROL_LENGTH, P.registered_lengths(c)[0]] for c in grid},
                eval_n=200, guided_n=0, guided_lengths={}, device=device, fmt=fmt)
        stages.append({"stage": name, "steps": stage_steps, "n_docs": len(docs),
                       "mix": weights, "final_loss": run["final_loss"],
                       "loss_curve": [(int(s), float(v)) for s, v in run.get("loss_curve", [])],
                       "train_s": round(time.time() - t0), "eval": ev, "guided": gv})
    del model
    torch.cuda.empty_cache()
    return stages


def run_grid(a):
    """The full three-cell grid under the bounded pad, on the registered guided grid."""
    specs = E.three_cell_specs(P.TRAIN_LENGTHS)
    world, r = TK.build_world(specs["composed"])
    tok = Tokenizer.build([world], r)
    ml = P.matched_lengths(tok)
    gg = P.guided_grid(ml)
    gg["composed"] = list(P.GUIDED_LENGTHS)
    grid = {c: sorted(set(list(P.registered_lengths(c)) + list(gg.get(c, ()))
                          + [P.CONTROL_LENGTH])) for c in P.LOCAL_CELLS}
    pad = PAD_WIDTH[a.format]
    floors = bounded_floors_for(gg, a.guided_n, pad=pad)
    out = {"generated": datetime.now(timezone.utc).isoformat(), "pad_width": pad,
           "format": a.format, "cfg": {**{k: v for k, v in vars(a).items()},
                                       "guided_grid": gg, "grid": grid,
                                       "k": specs["composed"].k},
           "floors": floors, "runs": []}
    ckpt_dir = a.ckpt_dir or f"{a.out_prefix}_ckpt"
    jl = open(f"{a.out_prefix}.jsonl", "a")
    for arch in a.archs.split(","):
        for seed in a.seeds:
            print(f"\n=== {arch} seed {seed} [pad {pad} / {a.format}] ===", flush=True)
            stages = grid_run_one(arch, seed, specs, tok, world, grid, steps=a.steps,
                                  batch=a.batch, d_model=a.d_model, n_layers=a.n_layers,
                                  n_heads=a.n_heads, lr=a.lr, train_n=a.train_n,
                                  eval_n=a.eval_n, guided_n=a.guided_n, guided_lengths=gg,
                                  device=a.device, fmt=a.format,
                                  loss_log_interval=a.loss_log_interval,
                                  guided_batch=a.guided_batch, ckpt_dir=ckpt_dir,
                                  answer_docs=a.pad_answer_docs)
            row = {"arch": arch, "seed": seed, "stages": stages}
            out["runs"].append(row)
            jl.write(json.dumps(row) + "\n")
            jl.flush()
            with open(f"{a.out_prefix}.json", "w") as f:
                json.dump(out, f, indent=1, default=float)
    jl.close()
    print(f"\nwrote {a.out_prefix}.json")


def report(path):
    """The three cells per seed against the BOUNDED-PAD floors, on both protocols.

    Per-seed values only — this family is bimodal at the emergence threshold and a mean over one
    converged and two floored seeds is a number no seed produced. The composed cell's column is a
    FLOOR comparison here and not a within-run direction, which is the whole point of the width:
    under the shipped format that column had no floor to be read against.

    The DENSE control's guided numbers are not reproduced here; they are the shipped run's
    (``results/s5bind_v3_three_cell_depthmatched_20260801.json``), at the same arch, size, steps,
    schedule and seeds.
    """
    res = json.load(open(path))
    fl, cfg = res["floors"], res["cfg"]
    gg = cfg["guided_grid"]
    n = cfg["guided_n"]
    print(f"\npad {res['pad_width']} / format {res['format']}   guided n={n}   "
          f"CLEARS = z>{P.Z_CLEAR} and margin>={P.MARGIN}\n")
    hdr = [f"{c}@{L}" for c in ("state", "bind", "composed") for L in gg.get(c, ())]
    print(f"{'arch':11s} {'sd':>2s} " + " ".join(f"{h:>14s}" for h in hdr))
    for r in res["runs"]:
        g = r["stages"][-1]["guided"]
        cells = []
        for c in ("state", "bind", "composed"):
            for L in gg.get(c, ()):
                blk = g.get(c, {}).get(str(L)) or {}
                v, sa = blk.get("match"), blk.get("slot_acc")
                f = (fl.get(f"{c}@{L}") or {}).get("floor")
                mark = "*" if v is not None and P.clears(v, f, n)[0] else " "
                cells.append("—" if v is None else f"{v:.3f}{mark}/{sa:.2f}")
        print(f"{r['arch']:11s} {r['seed']:2d} " + " ".join(f"{c:>14s}" for c in cells))
    frow = []
    for c in ("state", "bind", "composed"):
        for L in gg.get(c, ()):
            f = (fl.get(f"{c}@{L}") or {}).get("floor")
            frow.append("unfloorable" if f is None else f"{f:.4f}")
    print(f"{'floor':11s} {'':2s} " + " ".join(f"{c:>14s}" for c in frow))
    print("\ncell = guided match, '*' clears its bounded-pad floor, / per-token pad accuracy\n")
    print(f"{'arch':11s} {'sd':>2s}  PLAIN read")
    for r in res["runs"]:
        ev = r["stages"][-1]["eval"]
        row = []
        for c in ("state", "bind", "composed"):
            for L in cfg["grid"][c]:
                v = ev.get(c, {}).get(str(L))
                if v is not None:
                    row.append(f"{c[0]}{L}={v:.3f}")
        print(f"{r['arch']:11s} {r['seed']:2d}  " + " ".join(row))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--grid", action="store_true")
    ap.add_argument("--report", default=None, help="print the tables from a results JSON and exit")
    ap.add_argument("--decode_from", default=None,
                    help="CKPT_DIR: score saved weights on --decode_cells and exit")
    ap.add_argument("--decode_cells", default="composed@48,composed@64,composed@96")
    ap.add_argument("--attribute", default=None,
                    help="CKPT_DIR: attribute the guided answers of saved weights and exit")
    ap.add_argument("--pad_answer_docs", action="store_true",
                    help="add an answer-masked copy of each pad document to the mix")
    ap.add_argument("--ckpt_dir", default=None)
    ap.add_argument("--archs", default="gdp_hybrid")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--formats", default="dense,moved2,delta2")
    ap.add_argument("--format", default="moved2")
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--d_model", type=int, default=768)
    ap.add_argument("--n_layers", type=int, default=8)
    ap.add_argument("--n_heads", type=int, default=6)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--train_n", type=int, default=80000)
    ap.add_argument("--eval_n", type=int, default=1000)
    ap.add_argument("--guided_n", type=int, default=128)
    ap.add_argument("--guided_batch", type=int, default=128)
    ap.add_argument("--pilot_lengths", default="17,80")
    ap.add_argument("--loss_log_interval", type=int, default=500)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out_prefix", default="results/s5bind_v3_bounded_pad_pilot_20260802")
    a = ap.parse_args()
    if a.report:
        return report(a.report)
    if a.decode_from:
        cells = [(c.split("@")[0], int(c.split("@")[1])) for c in a.decode_cells.split(",")]
        got = decode_cells(a.decode_from, a.archs.split(","), a.seeds, cells, a.guided_n,
                           a.device, a.format, batch=a.guided_batch)
        with open(f"{a.out_prefix}_decode.json", "w") as f:
            json.dump(got, f, indent=1, default=float)
        print(f"wrote {a.out_prefix}_decode.json")
        return
    if a.attribute:
        sp = E.three_cell_specs(P.TRAIN_LENGTHS)
        for arch in a.archs.split(","):
            for seed in a.seeds:
                pth = E.checkpoint_path(a.attribute, arch, seed)
                if not Path(pth).exists():
                    print(f"  -- no checkpoint {pth}; skipped")
                    continue
                for cell, L in (("state", 17), ("state", 80), ("bind", 31), ("composed", 48)):
                    got = attribute_answers(pth, sp[cell], L, a.guided_n, a.device, a.format,
                                            batch=a.guided_batch)
                    print(f"  {arch} s{seed} {cell}@{L}: " + "  ".join(
                        f"{k}={v:.3f}" for k, v in got.items() if k != "n")
                        + f"  (n={got['n']})", flush=True)
        return
    if a.grid:
        return run_grid(a)

    specs = E.three_cell_specs(P.TRAIN_LENGTHS)
    world, r = TK.build_world(specs["composed"])
    tok = Tokenizer.build([world], r)
    lengths = [int(x) for x in a.pilot_lengths.split(",")]
    formats = [x for x in a.formats.split(",") if x]

    out = {"generated": datetime.now(timezone.utc).isoformat(),
           "pad_width": {f: PAD_WIDTH[f] for f in formats}, "formats": formats,
           "cfg": {k: v for k, v in vars(a).items()},
           "rows": []}
    Path(a.out_prefix).parent.mkdir(parents=True, exist_ok=True)
    jl = open(f"{a.out_prefix}.jsonl", "a")
    for fmt in formats:
        for arch in a.archs.split(","):
            for seed in a.seeds:
                print(f"\n=== pilot {arch} seed {seed} format {fmt} ===", flush=True)
                row = pilot_one(arch, seed, fmt, specs["state"], tok, world, steps=a.steps,
                                batch=a.batch, d_model=a.d_model, n_layers=a.n_layers,
                                n_heads=a.n_heads, lr=a.lr, train_n=a.train_n,
                                eval_n=a.eval_n, guided_n=a.guided_n, guided_lengths=lengths,
                                device=a.device, guided_batch=a.guided_batch,
                                loss_log_interval=a.loss_log_interval,
                                answer_docs=a.pad_answer_docs)
                out["rows"].append(row)
                jl.write(json.dumps(row) + "\n")
                jl.flush()
                with open(f"{a.out_prefix}.json", "w") as f:
                    json.dump(out, f, indent=1)
    jl.close()
    print(f"\nwrote {a.out_prefix}.json")


if __name__ == "__main__":
    main()

"""s5_bind_v3: the floor as a PROFILE over live slots, and the surface family as a fitted ranker.

Everything is recomputed from the exact rendered items a cell scores, through
``factworld.composition``'s own parser (which shares no code with the sampler).

  PROFILE  each cell's floor over W, not as a number: the partial-carry family (carry P in full
           and j of the m holder cells) and the block-drop family (play all but a block), with
           the one-structure bound drawn across it. The bound admits j = 0 and nothing above it,
           which is the check that it is a rule and not a threshold placed after the fact.
  SURFACE  the state-free read the one-at-a-time sweep missed: the reference slot of the last
           swap naming the queried agent. Reported per branch, since only the SAME branch names
           an agent, with the gate on and off.
  RANKER   the whole surface candidate set at once — a multinomial logit over 25 per-candidate
           features, fitted on one sample and scored HELD-OUT on a disjoint one. A one-at-a-time
           sweep reports the max over its rules, which is a selection statistic and which missed
           a 1.41x rule; a fitted ranker scored out of sample is neither.

Run:  .venv/bin/python scripts/probe_s5bind_v3_floor_20260731.py [--n 1500] [--stage all]
"""
import argparse
import math
import os
import random
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from dataclasses import replace                                     # noqa: E402

from factworld import composition as C                              # noqa: E402
from factworld import validity as V                                 # noqa: E402
from factworld.tasks import CANONICAL, generate                     # noqa: E402

CELLS = (("s5_bind_v3", 128), ("s5_bind_v3", 192), ("s5_bind_v3", 256),
         ("s5_bind_local_v3", 48), ("s5_bind_local_v3", 64), ("s5_bind_local_v3", 96),
         ("s5_bind_v3_state", 256), ("s5_bind_v3_bind", 256),
         ("s5_bind_local_v3_state", 96), ("s5_bind_local_v3_bind", 96))
WIDTHS = (0.01, 0.02, 0.05, 0.10, 0.15, 0.25, 0.50)
POSITIONS = (0.0, 0.2, 0.4, 0.6, 0.8, 0.9, 0.95)


# --- the profile ---------------------------------------------------------------------------
def profile_block(name, L, n):
    spec = CANONICAL[name]
    ex = generate(spec, "test", n=n, length=L)
    k, m = spec.k, spec.n_objects_active
    ch = 1.0 / (k - 1)
    named = V.s5_bind_v3_is_named(ex)
    query = V.s5_bind_v3_query_kind(ex)
    ns, ng = V.s5_bind_v3_shape(ex)
    wt, st = V.s5_bind_v3_task_cost(k, m, ns, ng, named, query)
    smin = V.s5_bind_v3_task_cost_min(k, m, ns, ng, named, query)
    dep = V.s5_bind_v3_task_depth(k, m, ns, ng, named, query)
    bound = (f"depth <= {V.S5_BIND_V3_MAX_DEPTH} hop against the algorithm's {dep}, and S < {smin}"
             if named else f"W <= {V.one_structure_bound(k, m)}")
    fl = dict(V.s5_bind_v3_floors(ex, k, m))
    fl.update(V.s5_bind_v3_family_floors(ex, k, m, named, query))
    op = V.s5_bind_v3_operative_floor(fl, k, m, ns, ng, named, query)
    basis = V.s5_bind_v3_floor_basis(k, m, ns, ng, named, query)
    print(f"\n== {name}@L{L}  k={k} m={m} swaps={ns} gives={ng} query={query} n={n}")
    print(f"   task cheapest correct algorithm: W={wt} S={st} (per-item minimum {smin}) "
          f"depth={dep};  floor row bound {bound}")
    if basis == "chance":
        print(f"   OPERATIVE FLOOR {op:.4f} = INFORMED CHANCE. No REGISTERED row that reads the "
              f"item is admissible\n   here — the one that would bound the cell is its own "
              f"one-hop algorithm, which no admitted row\n   may pay for — and every admitted "
              f"member of the swept give-scan family is too short to reach\n   the write the "
              f"sampler pins into the window, so each resolves nothing at all.")
    else:
        print(f"   OPERATIVE FLOOR {op:.4f} ({op / ch:.2f}x chance), MEASURED")
    prof = V.s5_bind_v3_slot_profile(ex, k, m, named, query)
    axis = prof[0]["axis"] if prof else "W"
    print(f"   profile along {axis} ({'live slots' if axis == 'W' else 'steps'} — the resource "
          f"this cell separates on)")
    print(f"   {axis:>6}  {'acc':>7} {'xchance':>8}  {'admitted':>9}  policy")
    for r in prof:
        print(f"   {r[axis]:>6}  {r['acc']:>7.4f} {r['acc'] / ch:>8.2f}  "
              f"{'yes' if r['admitted'] else 'NO':>9}  {r['row']}")
    sb = V.s5_bind_v3_surface_bound(ex, k)
    if sb is not None:
        ns_, ng_ = V.s5_bind_v3_shape(ex)
        sp = V.s5_bind_v3_surface_price(k, m, ns_, ng_, named, query, sb["weights"])
        print(f"   fitted surface ranker (DIAGNOSTIC, admitted={sp['admitted']}: best "
              f"implementation W={sp['W']} vs {sp['W_max']}, S={sp['S']} vs {sp['S_max']}, "
              f"{sp['A']} per-candidate accumulators): held out {sb['held_out']:.4f} "
              f"({sb['held_out'] / ch:.2f}x) on {sb['n_held_out']} items, "
              f"fit on {sb['n_fit']}, in-sample {sb['in_sample']:.4f}")
    if not named:
        prof = V.s5_bind_v3_partial_carry_profile(ex, m)
        print("   partial-carry j = 0.." + str(m) + " (W = k+j+1), x chance:")
        print("     j    " + "".join(f"{j:>6}" for j in range(m + 1)))
        print("     x    " + "".join(f"{v / ch:>6.2f}" for v in prof))
        wp = V.s5_bind_v3_width_profile(ex, WIDTHS, POSITIONS)
        print("   block-drop width profile (best position at each width, W = k+m+1), x chance:")
        print("     w    " + "".join(f"{w:>6.2f}" for w in WIDTHS))
        print("     x    " + "".join(f"{wp[w] / ch:>6.2f}" for w in WIDTHS))
    if axis == "W":
        print("   THE W AXIS HAS NO FORCE UNDER A SCRATCHPAD PROTOCOL — a frontier model's own, or")
        print("   this repo's guided format, which requires both maps to be written out at every")
        print("   event. Every row above is then available to every policy and the number a score")
        print("   must clear is the TOP of this profile, not the admitted max; on this cell that")
        print("   top is the task, so the cell is UNFLOORABLE there on BOTH of that protocol's")
        print("   channels. The floor printed above is the PLAIN protocol's.")
    else:
        print("   THE DEPTH BOUND DOES BIND UNDER A SCRATCHPAD PROTOCOL: a pad substitutes for")
        print("   REGISTERS, not for CHAINING — it makes a truncated walk affordable, not correct")
        print("   — so the excluded rows above are what a score on this cell has to beat and the")
        print("   admitted max is what it has to clear, under either protocol.")


# --- the surface read ----------------------------------------------------------------------
def surface_block(name, L, n):
    base = CANONICAL[name]
    print(f"\n== {name}@L{L}  the last-naming-swap surface read, by branch")
    for tag, spec in (("as registered", base),
                      ("gate flipped", replace(base, q_no_surface=not base.q_no_surface))):
        ex = generate(spec, "test", n=n, length=L)
        ch = 1.0 / (spec.k - 1)
        tal = {}
        for e in ex:
            rec = C.read(e.prompt)
            if rec is None or rec["query"][0] != "state":
                continue
            tgt = rec["query"][1]
            for kd, t, ref, src in reversed(rec["events"]):
                if kd != C.SWAP or t != tgt:
                    continue
                pred = f"{ref}." if src != "B" else (
                    None if rec["B0"].get(ref) is None else f"{rec['B0'][ref]}.")
                d = tal.setdefault(src, [0, 0])
                d[0] += 1
                d[1] += int(pred is not None and pred == e.answer)
                break
        cells = []
        for src in sorted(tal):
            d, h = tal[src]
            acc = h / d
            z = (acc - ch) / math.sqrt(ch * (1 - ch) / d)
            lbl = {"P": "SAME", "B": "CROSS", "N": "named"}[src]
            cells.append(f"{lbl} {acc:.4f} ({acc / ch:.2f}x, z={z:+.2f}, n={d})")
        print(f"   q_no_surface={spec.q_no_surface!s:<5} {tag}: " + "  ".join(cells))


# --- the fitted ranker ---------------------------------------------------------------------
FEATURES = (
    "stated_answer", "echo", "same_ref", "same_ref_1hop", "same_ref_2hop",
    "cross_ref_holder", "cross_ref_holder_1hop", "named_partner", "named_partner_1hop",
    "stated_preimage", "stated_2hop", "prev_naming_swap_slot", "in_last_swap_sentence",
    "n_named_first", "n_ref_slot", "n_give_ref", "last_mention_pos", "first_mention_pos",
    "last_same_ref_anywhere", "last_swap_named_anywhere", "last_give_stated_holder",
    "last_fact_agent", "mention_count", "agent_index", "end_distance",
)


def _feats(rec, agents):
    """Per-candidate surface features for one item. Nothing here carries a map: every feature is
    a stated-block read, a count, or a position, all at O(1) live slots."""
    P0, B0, evs = rec["P0"], rec["B0"], rec["events"]
    q = rec["query"][1]
    inv0 = {v: a for a, v in P0.items()}
    L = max(1, len(evs))
    naming = [(i, ref, src) for i, (kd, t, ref, src) in enumerate(evs)
              if kd == C.SWAP and t == q]
    ls = naming[-1] if naming else None
    ls2 = naming[-2] if len(naming) > 1 else None

    def slot(entry):
        if entry is None:
            return None
        _i, ref, src = entry
        return B0.get(ref) if src == "B" else ref

    n_named, n_ref, n_give_ref, last_pos, first_pos, cnt = ({a: 0 for a in agents} for _ in
                                                            range(6))
    for a in agents:
        last_pos[a], first_pos[a] = -1, -1
    for i, (kd, t, ref, src) in enumerate(evs):
        touched = [t] if kd == C.SWAP else []
        if kd == C.SWAP:
            n_named[t] = n_named.get(t, 0) + 1
        tok = ref if src != "B" else None
        if tok in n_ref:
            (n_ref if kd == C.SWAP else n_give_ref)[tok] += 1
            touched.append(tok)
        for a in touched:
            if a in cnt:
                cnt[a] += 1
                last_pos[a] = i
                if first_pos[a] < 0:
                    first_pos[a] = i
    last_swap = next(((t, ref, src) for kd, t, ref, src in reversed(evs) if kd == C.SWAP),
                     None)
    last_same = next((ref for kd, _t, ref, src in reversed(evs)
                      if kd == C.SWAP and src == "P"), None)
    last_give = next((ref for kd, _t, ref, src in reversed(evs) if kd == C.GIVE), None)
    lg_holder = B0.get(last_give) if last_give in B0 else last_give
    slot_ls, slot_ls2 = slot(ls), slot(ls2)
    src_ls = ls[2] if ls else None
    out = {}
    for j, c in enumerate(agents):
        out[c] = [
            float(c == P0.get(q)), float(c == q),
            float(src_ls == "P" and c == slot_ls),
            float(src_ls == "P" and c == P0.get(slot_ls)),
            float(src_ls == "P" and c == P0.get(P0.get(slot_ls))),
            float(src_ls == "B" and c == slot_ls),
            float(src_ls == "B" and c == P0.get(slot_ls)),
            float(src_ls == "N" and c == slot_ls),
            float(src_ls == "N" and c == P0.get(slot_ls)),
            float(c == inv0.get(q)), float(c == P0.get(P0.get(q))),
            float(slot_ls2 is not None and c == slot_ls2),
            float(last_swap is not None and c in (last_swap[0], last_swap[1])),
            n_named.get(c, 0) / L, n_ref.get(c, 0) / L, n_give_ref.get(c, 0) / L,
            last_pos[c] / L, first_pos[c] / L,
            float(c == last_same), float(last_swap is not None and c == last_swap[0]),
            float(c == lg_holder), float(c == list(P0)[-1] if P0 else False),
            cnt[c] / L, j / max(1, len(agents)), (L - 1 - last_pos[c]) / L,
        ]
    return out


def _softmax_fit(rows, golds, n_feat, epochs=400, lr=0.5, l2=1e-3, seed=0):
    w = [0.0] * n_feat
    rng = random.Random(seed)
    order = list(range(len(rows)))
    for ep in range(epochs):
        rng.shuffle(order)
        grad = [0.0] * n_feat
        for i in order:
            cand = rows[i]
            zs = [sum(a * b for a, b in zip(w, f)) for _c, f in cand]
            mx = max(zs)
            es = [math.exp(z - mx) for z in zs]
            tot = sum(es)
            for (c, f), e in zip(cand, es):
                p = e / tot
                y = 1.0 if c == golds[i] else 0.0
                for j in range(n_feat):
                    grad[j] += (y - p) * f[j]
        step = lr / len(rows)
        for j in range(n_feat):
            w[j] += step * (grad[j] - l2 * w[j])
        del ep
    return w


def _rank_acc(w, rows, golds):
    hit = 0
    for cand, g in zip(rows, golds):
        best = max(cand, key=lambda cf: sum(a * b for a, b in zip(w, cf[1])))
        hit += int(best[0] == g)
    return hit / max(1, len(rows))


def ranker_block(name, L, n):
    spec = CANONICAL[name]
    ex = generate(spec, "test", n=2 * n, length=L)
    agents = None
    rows, golds = [], []
    for e in ex:
        rec = C.read(e.prompt)
        if rec is None or rec["query"][0] != "state":
            continue
        agents = sorted(rec["P0"], key=lambda s: int(s[1:]))
        f = _feats(rec, agents)
        rows.append([(c, f[c]) for c in agents])
        golds.append(e.answer.strip().rstrip("."))
    if not rows:
        return
    half = len(rows) // 2
    w = _softmax_fit(rows[:half], golds[:half], len(FEATURES))
    tr = _rank_acc(w, rows[:half], golds[:half])
    ho = _rank_acc(w, rows[half:], golds[half:])
    ch = 1.0 / (spec.k - 1)
    se = math.sqrt(ch * (1 - ch) / max(1, len(rows) - half))
    print(f"\n== {name}@L{L}  fitted surface ranker, {len(FEATURES)} features, "
          f"fit on {half} items, scored on {len(rows) - half} disjoint")
    print(f"   in-sample {tr:.4f} ({tr / ch:.2f}x)   HELD-OUT {ho:.4f} ({ho / ch:.2f}x, "
          f"z={(ho - ch) / se:+.2f})   chance {ch:.4f}")
    top = sorted(zip(FEATURES, w), key=lambda t: -abs(t[1]))[:8]
    print("   largest weights: " + ", ".join(f"{nm} {v:+.2f}" for nm, v in top))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1500)
    ap.add_argument("--stage", default="all",
                    choices=("all", "profile", "surface", "ranker"))
    a = ap.parse_args()
    if a.stage in ("all", "profile"):
        for name, L in CELLS:
            profile_block(name, L, a.n)
    if a.stage in ("all", "surface"):
        for name, L in CELLS:
            if CANONICAL[name].query_arm == "state":
                surface_block(name, L, a.n)
    if a.stage in ("all", "ranker"):
        for name, L in CELLS:
            if CANONICAL[name].query_arm == "state":
                ranker_block(name, L, a.n)


if __name__ == "__main__":
    main()

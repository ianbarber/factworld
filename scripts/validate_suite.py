"""Validity gate over the CANONICAL task suite — guarantees cover the headline tasks.

For every canonical task, certify that no shallow baseline clears floor on the held-out test split:
  - oracle-consistency: gold answers resolve through the symbolic oracle (true by construction; we assert
    the answer span is well-formed and the answer-token type is as expected).
  - answer balance: the majority-answer baseline ≈ floor (no dominant-class shortcut).
  - recency shortcut: predicting the last in-prompt token of the answer's type (e.g. the final-give agent)
    stays near floor — the composition tasks are not solvable by "copy the most recent entity".
  - STRONG recency shortcut (binding/composite only): the full-answer heuristic "last give-event's
    recipient" (+ "that holder's stated a0 fact" for composite) — see factworld.validity. Every
    registered binding/composite task uses the last_write_uniform (v2) sampler, so this baseline is
    GATED (must stay near floor). The recency-defective v1 family — where this heuristic scored
    ~0.34@L16 on composite_copy_v1 / ~0.4 on binding_v1 — is RETIRED (tasks.RETIRED, issue #11):
    excluded from the suite run here; its known-shortcut annotation lives on the RETIRED dict.
  - SHALLOW-ADVERSARY floor (chain / s5_chain): the largest of the registered pointer-map
    adversaries — initial-map chase, initial-ref resolution, echo (factworld.validity), i.e. the
    cell's operative floor with the two chance rows removed, since chance is not a shortcut.
    The initial-map backhop is measured by s5_chain_floors but is not registered and so is not
    in this column: it is an unnamed member of a fixed-offset family whose accuracies sum to 1,
    so its null is uniform-over-non-start and a max over it measures selection.
  - SHALLOW-ADVERSARY floor (s5_bind): the largest registered mutual-reference policy
    (factworld.validity.S5_BIND_ADVERSARIES) — the coupling-blind rows, the zero-state pin
    chain, and the stated/one-hop rows. Registration is decided by RESOURCE CLASS: a row may set
    a floor only if it carries no structure-sized state (W = O(1) live slots), because the task's
    own cheapest correct algorithm carries P, its inverse and B. That rule is what excludes the
    block-drop family — window_f keeps the last f*L events, prefix_f the first f*L, and both are
    positions of one continuum whose members all carry both maps. Those rows are measured and
    printed as diagnostics, marked '†'. The SOURCE-STRUCTURE rung (s5_bind_v3) replaces the
    slots-only half with a ONE-STRUCTURE BOUND — a row may set a floor only if it holds at most
    one structure (W <= max(k,m)+1 under the W convention stated in factworld.validity) and pays
    no more steps than the task — which closes the partial-carry continuum (carry P in full and
    j of the m holder cells) by the same argument that closes the block-drop one. Its COMPONENT
    cells, whose own algorithm holds no structure at all, take the same move on the axis they
    separate on: at most ONE HOP composed, and strictly fewer steps than that algorithm's
    MINIMUM per-item cost. That excludes the truncated-carrier-walk continuum in both of its
    parameterisations and leaves the retrieval component only rows too short to reach the write
    the sampler pins into [L/10, 0.75L]. Every row is printed for every cell in the floor blocks
    below the table, and each cell's floor is then printed AS A PROFILE — over W on a composed
    cell, over STEPS on a component one, with both swept families (truncated walk, truncated
    give-scan) plotted on it, since a single number hides exactly the continuum that has to be
    visible. The surface family enters as a FITTED RANKER scored out of sample rather than as a
    max over one-at-a-time rules, which is a selection statistic.
A task PASSES if majority, recency, first-position and (where defined) strong-recency accuracy are
all well below 0.5 (near the 1/#answers floor). The pointer-map families answer over a much larger
space than 2, so their column is gated against their own chance level instead: no shallow policy
may reach twice 1/#answers. That gate reads the registered spec at its longest eval length; a
rescaled cell (the local sweep runs k=4..8 at L=4..8) carries its own floor rows, and a score
there is read against that cell's operative floor rather than against chance.

  .venv/bin/python scripts/validate_suite.py
"""
import os
import sys
from collections import Counter

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from factworld import tasks as TK          # noqa: E402
from factworld.render import Renderer, classify  # noqa: E402  (atomic-token type by prefix: g/v/r/o/...)
from factworld.validity import (  # noqa: E402
    S5_BIND_ADVERSARIES,
    S5_BIND_V3_CHANCE_ROWS,
    S5_BIND_V3_ROWS,
    S5_BIND_V3_TRUNCATION_ROWS,
    s5_bind_v3_admits,
    s5_bind_v3_classify,
    s5_bind_v3_family_floors,
    s5_bind_v3_family_rows,
    s5_bind_v3_floor_basis,
    s5_bind_v3_floors,
    s5_bind_v3_operative_floor,
    s5_bind_v3_partial_carry_profile,
    s5_bind_v3_query_kind,
    s5_bind_v3_shape,
    s5_bind_v3_slot_profile,
    s5_bind_v3_surface_bound,
    s5_bind_v3_width_profile,
    s5_bind_v3_is_named,
    S5_BIND_CHANCE_ROWS,
    S5_BIND_MAP_CARRYING_ROWS,
    S5_BIND_ROWS,
    S5_BIND_TRUNCATION_ROWS,
    S5_CHAIN_ADVERSARIES,
    S5_CHAIN_CHANCE_ROWS,
    comm_shallow_accuracy,
    s5_bind_floors,
    s5_bind_operative_floor,
    s5_chain_floors,
    strong_recency_accuracy,
)

N = 500
# THE OPERATIVE FLOOR IS A MAX OVER ROWS, so at a finite n it carries an upward selection bias of
# order the largest row's standard error even when every row sits at chance. At n = 500 and k = 12
# one row's standard error alone is 0.14 of chance, and that is what the component cells'
# published 1.30x and 1.08x were: ``last_write_1hop`` reads 1.30x at n = 500 on
# s5_bind_v3_state@256 and 0.98x at n = 4000. The floor number is therefore re-measured on a
# larger held-out sample, and only for the rows the rule ADMITS — the deep excluded walks are
# diagnostics and stay at n = 500, where they are already an order of magnitude off chance. The
# fitted surface ranker keeps its n = 500 FIT (the expensive half) and is scored on the large
# sample, since scoring a fitted ranker is a pass and the fit sample is what biases it down.
N_FLOOR = 4000

# The strong recency baseline only has a defined prediction on the give-stream families.
STRONG_REC_FAMILIES = ("binding", "composite")

# The commutative family gets its own four shallow adversaries (initial-only / last-turn-only /
# entity-blind-sum / count-mod-k, factworld.validity.comm_shallow_accuracy); the MAX of the four
# fills the strongrec-style column and folds into the verdict.
COMM_FAMILIES = ("commutative",)

# The pointer-map families get theirs (factworld.validity.s5_chain_floors); the MAX fills the
# same column. DERIVED, not restated: the registered adversaries minus the chance rows, since
# uniform and uniform-over-non-start are what a shortcut has to BEAT, not shortcuts. Registering
# a new adversary in factworld.validity therefore reaches this column with no edit here, and a
# row that cannot set a floor cannot enter the gate either.
S5_CHAIN_FAMILIES = ("s5_chain", "chain")
S5_CHAIN_SHORTCUTS = tuple(n for n in S5_CHAIN_ADVERSARIES if n not in S5_CHAIN_CHANCE_ROWS)

# The mutual-reference family, derived the same way: registering a row in factworld.validity is
# enough to put it in this column, and a row that cannot set a floor cannot enter the gate.
S5_BIND_FAMILIES = ("s5_bind",)
S5_BIND_SHORTCUTS = tuple(n for n in S5_BIND_ADVERSARIES if n not in S5_BIND_CHANCE_ROWS)
# The map-carrying rows are out of the floor by class, so they are checked instead of gated: a
# cell where one of them reads far off chance has a live policy the class rule would be hiding.
S5_BIND_DIAGNOSTIC_ROWS = S5_BIND_TRUNCATION_ROWS
S5_BIND_DIAGNOSTIC_MAX = 2.0                     # multiples of the informed chance 1/(k-1)

# The SOURCE-STRUCTURE rung. Which rows enter the gate is decided by the CLASS RULE
# (factworld.validity.s5_bind_v3_classify: at most one structure held on a composed cell, at most
# one hop chained and strictly under the algorithm's per-item minimum cost on a component one),
# evaluated at the cell's own shape — so registering a row in factworld.validity reaches this
# column with no edit here, and a row that cannot set a floor cannot enter the gate either. On a
# COMPOSED cell the class-excluded rows are printed as diagnostics and separately checked to sit
# within S5_BIND_DIAGNOSTIC_MAX of chance: the exclusion is a cost argument, and a live policy
# hiding behind it would show up there. On a COMPONENT cell that check is not available and must
# not be faked — the excluded end there IS the cell's own algorithm with a few events dropped and
# reads 9.3x chance by construction. What is checked instead is the ADMITTED end: the operative
# floor itself must sit within S5_BIND_DIAGNOSTIC_MAX of chance, on every cell.
S5_BIND_V3_SHORTCUTS = tuple(n for n in S5_BIND_V3_ROWS if n not in S5_BIND_V3_CHANCE_ROWS)


def positional_pred(prompt: str, ans_type: str, which: str):
    """The first/last token in the prompt whose type matches the answer's type — a fixed-POSITION shortcut.
    `which='last'` is the recency shortcut; `which='first'` catches 'the answer is always the first
    fact's value'. Operates on the normalized (detached-punctuation) form so attached-punctuation
    tokens like `v109.` / `g0's` are classified correctly."""
    toks = Renderer.normalize(prompt).split()
    it = reversed(toks) if which == "last" else toks
    for t in it:
        if classify(t) == ans_type:
            return t
    return None


def main():
    print(f"Validity gate over CANONICAL suite (n={N} held-out test, at eval_lengths[-1]; "
          f"RETIRED specs excluded — see tasks.RETIRED)\n")
    print(f"  {'task':<22} {'#ans':>5} {'floor':>6} {'majority':>9} {'recency':>8} {'firstpos':>9} {'strongrec':>10}   verdict")
    all_ok = True
    bind_rows = {}
    bind_v3_rows = {}
    for name, spec in TK.CANONICAL.items():
        test = TK.generate(spec, "test", n=N, length=spec.eval_lengths[-1])
        # normalize answers so the check is format-agnostic (attached `.` -> ` .`)
        ans_norm = [Renderer.normalize(e.answer) for e in test]
        firsts = [a.split()[0] for a in ans_norm]
        assert all(a.split()[-1] == "." for a in ans_norm), f"{name}: answer not '.'-terminated"
        atype = classify(firsts[0])                      # answer-token type (g/v/r)
        assert all(classify(f) == atype for f in firsts), f"{name}: inconsistent answer-token type"
        distinct = len(set(firsts))
        floor = 1.0 / distinct
        majority = Counter(firsts).most_common(1)[0][1] / N
        recency = sum(positional_pred(e.prompt, atype, "last") == f for e, f in zip(test, firsts)) / N
        firstpos = sum(positional_pred(e.prompt, atype, "first") == f for e, f in zip(test, firsts)) / N
        ok = majority < 0.5 and recency < 0.5 and firstpos < 0.5
        if spec.family in STRONG_REC_FAMILIES:
            # every registered give-stream task is a v2 (last_write_uniform) spec: GATED.
            assert spec.last_write_uniform, \
                f"{name}: non-uniform (v1) sampler in CANONICAL — v1 specs belong in RETIRED"
            strongrec = strong_recency_accuracy(test, spec.family)
            ok &= strongrec < 0.5
            srec_col = f"{strongrec:>10.3f}"
        elif spec.family in COMM_FAMILIES:
            # commutative rung: the strongest of the four dial-fold shallow adversaries.
            strongrec = max(comm_shallow_accuracy(test, spec.k_positions).values())
            ok &= strongrec < 0.5
            srec_col = f"{strongrec:>10.3f}"
        elif spec.family in S5_CHAIN_FAMILIES:
            # pointer-map rung: the strongest registered shallow policy, gated against this
            # task's own chance level. Rows the stream cannot support are absent, not zero
            # (no events -> no chase, no references -> no ref resolution).
            fl = s5_chain_floors(test, spec.k, has_events=(spec.family == "s5_chain"))
            strongrec = max([fl[n] for n in S5_CHAIN_SHORTCUTS if n in fl], default=0.0)
            ok &= strongrec < 2.0 * floor
            srec_col = f"{strongrec:>10.3f}"
        elif spec.family in S5_BIND_FAMILIES and spec.source_ablation:
            # source-structure rung: every registered policy, recomputed from these exact items,
            # classified by cost at this cell's own shape.
            m = spec.n_objects_active
            ns, ng = s5_bind_v3_shape(test)
            named = s5_bind_v3_is_named(test)
            query = s5_bind_v3_query_kind(test)
            fl = s5_bind_v3_floors(test, spec.k, m)
            # the two swept component families (truncated carrier walk, truncated give-scan)
            # carry each component cell's continuum; every member is classified on its own cost,
            # so they enter the floor exactly where the rule admits them and nowhere else.
            fam = s5_bind_v3_family_floors(test, spec.k, m, named, query)
            allfl = dict(fl)
            allfl.update(fam)
            cls = s5_bind_v3_classify(spec.k, m, ns, ng, named, query, rows=tuple(allfl))
            gated = [n for n in S5_BIND_V3_SHORTCUTS if n in fl and cls[n]]
            # the surface family enters as a FITTED bound scored out of sample, not as a max
            # over one-at-a-time rules — see validity.s5_bind_v3_surface_bound. The fit runs on
            # the scored split and the scoring on items N..N+N_FLOOR-1, which are disjoint from
            # it and deterministic, so the estimate is out of sample and the fit gets the whole
            # N. It is a W=2, one-hop, one-pass policy and is admitted by the same rule as any
            # row — on a component cell it pays 2L against an algorithm whose minimum is 2L+2.
            big = TK.generate(spec, "test", n=N + N_FLOOR,
                              length=spec.eval_lengths[-1])[N:]
            sb = s5_bind_v3_surface_bound(test, spec.k, held_out=big)
            if sb is not None and not s5_bind_v3_admits("surface_ranker", spec.k, m, ns, ng,
                                                        named, query):
                sb = None
            # the floor itself, re-measured at N_FLOOR on the rows the rule admits
            nsb, ngb = s5_bind_v3_shape(big)
            keep = tuple(r for r in s5_bind_v3_family_rows(spec.k, m, nsb, ngb, named, query)
                         if s5_bind_v3_admits(r, spec.k, m, nsb, ngb, named, query))
            bigfl = dict(s5_bind_v3_floors(big, spec.k, m))
            bigfl.update(s5_bind_v3_family_floors(big, spec.k, m, named, query, rows=keep))
            op = s5_bind_v3_operative_floor(bigfl, spec.k, m, nsb, ngb, named, query)
            if sb is not None and (op is None or sb["held_out"] > op):
                op = sb["held_out"]
            strongrec = op or 0.0
            ok &= strongrec < 0.5
            # AND the real gate on this rung: the admitted end of the profile must sit at
            # informed chance. A rule that admits a policy far above it has not closed.
            ok &= strongrec < S5_BIND_DIAGNOSTIC_MAX / max(1, spec.k - 1)
            srec_col = f"{strongrec:>10.3f}"
            # The class exclusion is a cost argument; on the COMPOSED cell a live policy hiding
            # behind it would show up in the truncation diagnostics, so they are checked there.
            # On a component cell a truncation that keeps the load-bearing end IS the component's
            # own algorithm at a discount, and it costs k + m slots against the component's 2.
            if not named:
                lim = S5_BIND_DIAGNOSTIC_MAX / max(1, spec.k - 1)
                ok &= all(fl[n] < lim for n in S5_BIND_V3_TRUNCATION_ROWS if n in fl)
            bind_v3_rows[name] = (fl, op, gated, spec.eval_lengths[-1], spec.k, sb,
                                  s5_bind_v3_floor_basis(spec.k, m, ns, ng, named, query),
                                  (test, m, named, query), fam, cls, bigfl)
        elif spec.family in S5_BIND_FAMILIES:
            # mutual-reference rung: every registered policy, recomputed from these exact items.
            fl = s5_bind_floors(test, spec.k)
            gated = [n for n in S5_BIND_SHORTCUTS if n in fl]
            strongrec = max([fl[n] for n in gated], default=0.0)
            ok &= strongrec < 0.5
            srec_col = f"{strongrec:>10.3f}"
            # the class-excluded rows are not a floor, but on a gated coupled cell they must be
            # dead: the chain gate is what earns the exclusion.
            if spec.coupled:
                lim = S5_BIND_DIAGNOSTIC_MAX / max(1, spec.k - 1)
                ok &= all(fl[n] < lim for n in S5_BIND_DIAGNOSTIC_ROWS if n in fl)
            bind_rows[name] = (fl, s5_bind_operative_floor(fl, coupled=spec.coupled), gated,
                               spec.eval_lengths[-1], spec.k)
        else:
            srec_col = f"{'—':>10}"
        all_ok &= ok
        print(f"  {name:<22} {distinct:>5} {floor:>6.3f} {majority:>9.3f} {recency:>8.3f} {firstpos:>9.3f} {srec_col}   {'PASS' if ok else 'FLAG'}")
    if bind_rows:
        # The mutual-reference floors sit far above chance, so every row is printed: the
        # operative floor (the max over all registered rows) is what a score is read against,
        # while the gate reads only the rows marked '*' — see the module docstring.
        print(f"\n  s5_bind registered floors (n={N} at eval_lengths[-1]; "
              f"'*' = enters the gate, '†' = carries a map, so it is a diagnostic and not a "
              f"floor, 'op' = the number a score is read against, "
              f"'op/ch' = op over the informed chance 1/(k-1))")
        print("    " + f"{'task':<24}{'L':>5}" + "".join(f"{r[:10]:>11}" for r in S5_BIND_ROWS)
              + f"{'op':>9}{'op/ch':>8}")
        for name, (fl, op, gated, L, k) in bind_rows.items():
            cells = "".join(
                (f"{fl[r]:>10.3f}"
                 f"{'*' if r in gated else '†' if r in S5_BIND_MAP_CARRYING_ROWS else ' '}"
                 if r in fl else f"{'—':>11}")
                for r in S5_BIND_ROWS)
            ratio = op / fl["uniform_non_initial"] if "uniform_non_initial" in fl else None
            print(f"    {name:<24}{L:>5}" + cells + f"{op:>9.3f}"
                  + (f"{ratio:>8.2f}" if ratio is not None else f"{'—':>8}"))
    if bind_v3_rows:
        print(f"\n  s5_bind_v3 registered floors (rows at n={N} at eval_lengths[-1], "
              f"'op' re-measured at n={N_FLOOR} on the admitted rows; "
              f"'*' = admitted by the class rule, so it enters the gate and may set the floor "
              f"('op' is the max over ALL admitted rows, the swept families included), "
              f"'†' = class-excluded — on a composed cell it holds both structures or pays more "
              f"steps than the task, on a component cell it composes more than one hop or pays "
              f"what that cell's own algorithm pays; 'op/ch' = op over the informed chance "
              f"1/(k-1), 'basis' = whether any admitted REGISTERED row is a measured policy)")
        print("    " + f"{'task':<26}{'L':>5}"
              + "".join(f"{r[:10]:>11}" for r in S5_BIND_V3_ROWS)
              + f"{'surf.fit':>10}{'op':>9}{'op/ch':>8}  basis")
        for name, (fl, op, gated, L, k, sb, basis, _ctx, _fam, _cls, _big) in bind_v3_rows.items():
            cells = "".join((f"{fl[r]:>10.3f}{'*' if r in gated else '†'}"
                             if r in fl else f"{'—':>11}") for r in S5_BIND_V3_ROWS)
            ratio = op / fl["uniform_non_initial"] if "uniform_non_initial" in fl else None
            sbc = f"{sb['held_out']:>10.3f}" if sb else f"{'—':>10}"
            print(f"    {name:<26}{L:>5}" + cells + sbc + f"{op:>9.3f}"
                  + (f"{ratio:>8.2f}" if ratio is not None else f"{'—':>8}")
                  + f"  {basis}")
        print(f"\n  s5_bind_v3 FLOOR PROFILES (n={N}). A cell's floor is not one number: it is "
              f"what each budget buys,\n  along the resource that cell separates on — LIVE SLOTS "
              f"on the composed cell (the one-structure bound\n  W <= max(k,m)+1 is the last "
              f"admitted row; everything above it holds both structures and is doing\n  the "
              f"composition at a discount) and STEPS on a component one, where the bound is ONE "
              f"HOP and the\n  excluded end is the cell's own algorithm with a few events "
              f"dropped, sitting one step under it.\n  THE W AXIS HAS NO FORCE IN THE FRONTIER "
              f"REGIME — a model with a scratchpad is not register-bounded,\n  so every row "
              f"below is available to it and the number a frontier score must clear is the TOP "
              f"of the\n  profile, not the admitted max. The DEPTH axis does bind there: a "
              f"scratchpad does not make a\n  truncated walk correct, it only makes it "
              f"affordable.")
        for name, (fl, op, gated, L, k, sb, basis, ctx, fam, cls, bigfl) in bind_v3_rows.items():
            test, m, named, query = ctx
            prof = s5_bind_v3_slot_profile(test, k, m, named, query)
            ch = 1.0 / max(1, k - 1)
            axis = prof[0]["axis"] if prof else "W"
            bound = ("depth <= 1 hop and steps < the algorithm's per-item minimum" if named
                     else f"W <= max(k,m)+1 = {max(k, m) + 1}")
            print(f"    {name}@L{L}  k={k} m={m}  chance {ch:.4f}  along {axis}, "
                  f"admitted while {bound}  floor {op:.4f} ({op / ch:.2f}x, {basis})")
            print("      " + "  ".join(
                f"{axis}={r[axis]}:{r['acc'] / ch:.2f}x{'' if r['admitted'] else '*'}"
                for r in prof))
            if not named:
                pc = s5_bind_v3_partial_carry_profile(test, m)
                wp = s5_bind_v3_width_profile(test)
                print("      partial-carry j=0.." + str(m) + " x chance: "
                      + " ".join(f"{v / ch:.2f}" for v in pc))
                print("      block-drop width x chance:  "
                      + " ".join(f"{w:.2f}:{v / ch:.2f}" for w, v in sorted(wp.items())))
            elif fam:
                # the component continuum, in the order the rule sees it, so the excluded member
                # one step under the cell's own algorithm is visible next to the admitted ones.
                print("      swept family x chance ('*' = excluded): " + " ".join(
                    f"{r.replace('trunc_walk_', '').replace('give_scan_', '')}"
                    f":{fam[r] / ch:.2f}{'' if cls[r] else '*'}"
                    for r in sorted(fam, key=lambda r: fam[r])))
            if sb is not None:
                print(f"      fitted surface ranker (W=2, {len(sb['weights'])} features, "
                      f"fit {sb['n_fit']} / held out {sb['n_held_out']}): "
                      f"{sb['held_out']:.4f} ({sb['held_out'] / ch:.2f}x), "
                      f"in-sample {sb['in_sample']:.4f}")
        print("    (* = not admitted. On a COMPOSED cell that means the row holds both "
              "structures. On a COMPONENT cell\n     the W bound is vacuous — its own algorithm "
              "already holds none — so the profile runs along STEPS\n     and a row is admitted "
              "only if it chains at most ONE hop and pays strictly less than that\n     "
              "algorithm's per-item MINIMUM. On the state component that excludes the whole "
              "truncated-walk\n     continuum, whose cheap end sits one step under the algorithm "
              "at 9.3x chance. On the retrieval\n     component the depth bound is vacuous too — "
              "that algorithm is one hop — and the cost bound admits\n     only budgets too "
              "short to reach the write the sampler pins into [L/10, 0.75L], every one of "
              "which\n     resolves nothing; the floor is informed chance and the basis column "
              "says 'chance' rather than\n     printing 1.00x as though a registered policy had "
              "been measured up to it.)")
    print(f"\nSUITE VALIDITY: {'PASS — no shallow/recency/position shortcut clears floor on any canonical task' if all_ok else 'FLAG — investigate'}")
    return all_ok


if __name__ == "__main__":
    sys.exit(0 if main() else 1)

# FactWorld — what this project is (anchor for agents and contributors)

FactWorld is a **composition instrument**. Recall is well tested by MQAR; state tracking by the
S5 word-problem literature. This instrument tests **both, independently and in composition** —
identically for frontier models over an API and for small from-scratch models, so it can (a)
place a new frontier model and (b) attribute capabilities to architectural and training choices.
Tasks render as natural language (so pretrained models can take them) over a constrained
vocabulary (so small-scale experiments can too).

## The taxonomy

| | Task | Notes |
|---|---|---|
| **Component: recall** | `recall_copy` | single-query, deferred-readout MQAR variant; pool = load axis |
| — parametric variant | `recall_v1` / `conflict_v1` | retrieval from weights (local models) |
| **Component: state tracking** | `binding` | last-write-wins (absorbing updates — NOT abelian group ops) |
| — commutative variant | `commutative_v1` | order-free per-entity accumulation mod k (every event matters, order does not); calibrated — reads only in the thinking regime (instant and d256-local at chance); experimental until a full roster run |
| — non-abelian variant | `s5` | order-sensitive permutation streams; length = sequence stress |
| **Composition: state × recall** | `composite` | the two-hop; headline statistic = **gap** (binding − composed) |
| **Composition: recall ∘ recall** | `chain` | pointer chase; depth axis at fixed breadth |

**Axes** (each tests a different thing): solve rate; pool/breadth (working-set load); depth/length
(iteration count); regime (**instant** = reasoning off + answer contract = in-weights, vs
**thinking** = generous budget); reasoning tokens needed to solve. Difficulty knobs are
**calibration parameters** — used to place each model class mid-scale, never published as axes.

## The thesis

**No element is free; each is paid for by an architectural or training choice.** Current price
table (local evidence): deferred recall ← product recurrence (not attention — the transformer
aces adjacent readout, fails deferred); adjacent recall ← attention; last-write state ←
recurrence; non-abelian state ← dense per-step supervision (to form) + recurrent hybrid (to
extrapolate); depth extrapolation ← **open**; local composition (value leg) ← **open**.

## The two finding sets

1. **Frontier profiles** (9 models, OpenRouter): per-axis scores + composition gap, two regimes.
   The instant and thinking rankings are near-orthogonal; present per-axis ranks and profiles,
   not a single scalar.
2. **Local attributions**: which architecture/training choice buys which component
   (transformer vs recurrent hybrids vs fprm; supervision density; curricula).

## Rules that keep the instrument honest

- Relaxed match is the canonical metric; exact/contains/last_n are diagnostics.
- Floors are first-class rows (recency heuristic; object-filter E[1/w]); scores are read against
  them. Marks are plain-language (†, *, ‡, ⊘ "not measurable at this budget").
- No "walls/horizons/knees/cliffs" in headlines; no capability-frontier mapping — the question is
  always "does this improve the composition measurement in either regime?"
- Reasoning-on cells need explicit large budgets and published empty rates; instant cells need the
  answer contract. Discriminate before spending: never buy cells predicted to sit at ceiling/floor.
- Tasks are versioned; a defective version is retired outright (see issue #11), never kept scored.

Primary docs: `reports/frontier-benchmark.md` (narrative), `docs/benchmark/results.md` (rendered),
`docs/experiments/README.md` (log), `factworld/benchmark.py` (registry).

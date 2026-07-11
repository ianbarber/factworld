# FactWorld — what this project is (anchor for agents and contributors)

FactWorld is a **composition instrument**. Recall is well tested by MQAR; state tracking by the
S5 word-problem literature. This instrument tests **both, independently and in composition** —
identically for frontier models over an API and for small from-scratch models, so it can (a)
place a new frontier model and (b) attribute capabilities to architectural and training choices.
Tasks render as natural language (so pretrained models can take them) over a constrained
vocabulary (so small-scale experiments can too).

## The three-part story

Both README.md and reports/frontier-benchmark.md open with part 1 and are structured in this
order:

1. **The instrument** (the core): certifying recall and state tracking independently and in
   composition — tasks, floors, marks, regimes, validity machinery.
2. **Evalling the frontier**: the benchmark built on the instrument — a clear view of current
   models (9 models, two regimes), with a cheap add-a-model path. The instant and thinking
   rankings are near-orthogonal; present per-axis ranks and profiles, never a single scalar.
3. **Exploring the architectures**: which components elicit each capability at small scale —
   transformers, recurrent hybrids (gdp/gdn), fprm; supervision density and curricula — at
   matched compute.

History lives in phases/ (linked as provenance) and docs/experiments/README.md (the archival
log); the main docs describe what the instrument is, with no work-in-progress language — a
correction appears only as a contained methodological note when it is scientifically
interesting.

## The taxonomy

| | Task | Notes |
|---|---|---|
| **Component: recall** | `recall_copy` | single-query, deferred-readout MQAR variant; pool = load axis |
| — parametric variant | `recall_v1` / `conflict_v1` | retrieval from weights (local models) |
| **Component: state tracking** | `binding` | last-write-wins (absorbing updates — NOT abelian group ops) |
| — commutative variant | `commutative_v1` | order-free per-entity accumulation mod k (every event matters, order does not); experimental — discriminates in the thinking regime only (instant and d256-local at chance) |
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

## Rules that keep the instrument honest

- Relaxed match is the canonical metric; exact/contains/last_n are diagnostics.
- Floors are first-class rows (recency heuristic; object-filter E[1/w]); scores are read against
  them. Marks are plain-language (†, *, ‡, ⊘ "not measurable at this budget").
- No "walls/horizons/knees/cliffs" in headlines; no capability-frontier mapping — the question is
  always "does this improve the composition measurement in either regime?"
- Reasoning-on cells need explicit large budgets and published empty rates; instant cells need the
  answer contract. Discriminate before spending: never buy cells predicted to sit at ceiling/floor.
- Tasks are versioned; a defective version is retired outright, never kept scored.

Primary docs: `reports/frontier-benchmark.md` (narrative), `docs/benchmark/results.md` (rendered),
`docs/experiments/README.md` (log), `factworld/benchmark.py` (registry).

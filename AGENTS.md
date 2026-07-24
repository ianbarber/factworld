# FactWorld — what this project is (anchor for agents and contributors)

FactWorld is a **composition instrument**. Recall is well tested by MQAR; state tracking by the
S5 word-problem literature. This instrument tests **both, independently and in composition** —
identically for frontier models over an API and for small from-scratch models. Tasks render as
natural language (so pretrained models can take them) over a constrained vocabulary (so
small-scale experiments can too).

**The instrument is the headline.** Findings — that composition shows up differently by regime,
that it can be bought architecturally in different ways, that some tasks need reasoning — are
interesting but downstream. Every task in the suite answers to two questions:

1. **Does it differentiate frontier models well?**
2. **Can we explore it architecturally?**

A task that does neither is calibration work or dead weight, and does not appear in the docs.

## The three outputs

- **The repo** — a project for evaluating and exploring models and architectures. `README.md`
  carries the project sense: what the instrument is, the taxonomy, how to use it, where results
  live.
- **The report** (`reports/factworld-consolidated.md`) — the single narrative: why the
  instrument is interesting, with the current findings. Practitioner-oriented, results-bearing,
  terse. There is exactly one report.
- **FactWorldBench** — a frontier-model ranking derived from the instrument, published
  externally. Its feed is the generated `docs/benchmark/` page (`scripts/render_benchmark.py` →
  results.md / results.csv / index.html / figures, all regenerated from
  `results/benchmark/history.jsonl`). Protocol machinery (marks, budgets, per-cell provenance)
  lives there, not in the report.

History lives in phases/ (linked as provenance) and docs/experiments/README.md (the archival
log).

## The taxonomy

| | Task | Notes |
|---|---|---|
| **Component: recall** | `recall_copy` | single-query, deferred-readout MQAR variant; pool = load axis |
| — parametric variant | `recall_v1` / `conflict_v1` | retrieval from weights (local models) |
| **Component: state tracking** | `binding` | last-write-wins (absorbing updates — NOT abelian group ops) |
| — commutative variant | `commutative_v1` | order-free per-entity accumulation mod k (every event matters, order does not); experimental — reads in the thinking regime only, and the roster run failed the pre-registered promotion bar (only gpt-5.5 CI-separates); instant and d256-local answer-only at chance (per-step traces form it in-distribution for the recurrent archs) |
| — non-abelian variant | `s5` | order-sensitive permutation streams; length = sequence stress |
| **Composition: state × recall** | `composite` | the two-hop; the **gap** (binding − composed) is its derived statistic |
| **Composition: recall ∘ recall** | `chain` | pointer chase; depth axis at fixed breadth |
| **Composition: non-abelian state × serial dereference** | `s5_chain` | the FactWorldBench headline task — track a pointer map through order-sensitive events, then dereference it 8 hops deep; items gated so echo/fixed-hop heuristics score exactly 0 |

**Axes** (each tests a different thing): solve rate; pool/breadth (working-set load); depth/length
(iteration count); regime (**instant** = reasoning off + answer contract = in-weights, vs
**thinking** = generous budget); reasoning tokens needed to solve. Difficulty knobs are
**calibration parameters** — used to place each model class mid-scale, never published as axes.

## Findings so far (downstream of the instrument)

**No element is free; each is paid for by an architectural or training choice.** Current price
table (local evidence): deferred recall ← product recurrence (not attention — the transformer
aces adjacent readout, fails deferred); adjacent recall ← attention; last-write state ←
recurrence; non-abelian state ← dense per-step supervision (to form) + recurrent hybrid (to
extrapolate); commutative state ← dense per-step supervision + recurrence (formation only);
depth extrapolation ← **open**; local composition (value leg) ← **open**. Over the API,
composition is rented at inference: reasoning tokens buy it, monotonically with effort, and the
instant regime shows what survives in weights.

## Rules that keep the instrument honest

- **Docs state findings only.** No work-in-progress language, no status notes, no accounting for
  absent rows or sections — absence needs no apology. A negative result is publishable only when
  it discriminates; a null at an uncalibrated operating point is unfinished calibration and does
  not appear. Status lives in issues and PR bodies. A correction appears only as a contained
  methodological note when it is scientifically interesting.
- One metric, one name: the canonical evaluator is **match** — strip a trailing period from both
  sides and compare the model's first len(gold) whitespace tokens to the gold answer; binary per
  item, no partial credit (`factworld.tasks.score_relaxed`). Containment is the one published
  diagnostic. Stored record keys keep their historical names (metrics.relaxed/exact/contains/
  last_n) — a presentation convention, never a history rewrite.
- Symmetric contamination policy: ⊘ = not measurable at this budget; ≤x† = upper bound, covert
  reasoning on most calls (rtok on > 50% of the canonical attempt's calls); neither participates
  in orderings — not in figure sorts, not in cross-model ordering prose.
- Floors are first-class rows (recency heuristic; object-filter E[1/w]); scores are read against
  them, and both are recomputed at render time from the exact deterministic items, so they are
  independently checkable. Marks are plain-language (†, *, ‡, ⊘ "not measurable at this budget",
  —ᶠ "gap not interpretable: binding at the floor").
- No "walls/horizons/knees/cliffs" in headlines; no capability-frontier mapping — the question is
  always "does this improve the composition measurement in either regime?"
- Reasoning-on cells need explicit large budgets and published empty rates; instant cells need the
  answer contract. Discriminate before spending: never buy cells predicted to sit at ceiling/floor.
- Tasks are versioned; a defective version is retired outright, never kept scored.

Primary docs: `reports/factworld-consolidated.md` (the report), `docs/benchmark/results.md`
(rendered bench feed), `docs/experiments/README.md` (log), `factworld/benchmark.py` (registry).

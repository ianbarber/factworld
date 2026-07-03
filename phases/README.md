# Phases — prior work, archived

FactWorld is being reframed around a single natural-language evaluation and one
consolidated technical report (in progress). The earlier standalone papers and
their reproduction kits are preserved here as discrete **phases**. Each phase is
self-contained: its report, its per-claim result tables, and the exact scripts
that produced them.

These phases were run on the earlier *atomic-token* ("v1") rendering format.
That format lives in git history; the current benchmark renders clean natural
language (see `factworld/render.py`). Conclusions that are format-independent
(recall × state-tracking × composition ordering, the supervision-density lever,
architecture families) carry over; absolute numbers do not and are being
re-measured.

## Phase 1 — the instrument

[`01-instrument/factworld.md`](01-instrument/factworld.md) ·
[`01-instrument/factworld.pdf`](01-instrument/factworld.pdf)

*FactWorld: An Oracle-Validated Instrument for Composing Recall, State-Tracking,
and Knowledge.* Defines the frozen, versioned task suite, the symbolic oracle,
the no-leak render↔parse contract, and the position-strict exact-match metric.
The validity gate (`scripts/validate_suite.py`) still certifies the current
suite.

## Phase 2 — non-abelian state-tracking recipe

[`02-non-abelian-state/report.md`](02-non-abelian-state/report.md) ·
[`02-non-abelian-state/report.pdf`](02-non-abelian-state/report.pdf) ·
reproduction kit: [`02-non-abelian-state/REPRODUCE.md`](02-non-abelian-state/REPRODUCE.md)

*FactWorld: A Recipe for Length-Generalizing Non-Abelian State-Tracking.* Shows
that on `s5_v1` the bottleneck is supervision density and training distribution,
not architecture or parameter count: the same GDP backbone that floors under
sparse outcome supervision solves the task under dense per-step state
supervision, and a clean circuit can be extended to ~8× the trained horizon with
no target-length labels. Every claim maps to one script in this directory.

## What the current reframing carries forward

- The **composition probe is the crux.** Decomposing `composite_*` answers into
  the *holder* leg (state-tracking) and the *value* leg (in-context recall) shows
  that failures are overwhelmingly "holder right, value wrong" — i.e. the 2-hop
  routing, not the individual capabilities. The runner now reports this
  decomposition directly.
- **Format is a real but separable variable.** Clean natural language with one
  phrasing per statement type and attached punctuation roughly doubles
  composition convergence versus the atomic-token format at small scale; for
  pretrained chat models the dominant variable remains explicit output-format
  instructions.
- **Architecture still matters on learnable tasks.** Recurrent hybrids (GDP) and
  the weight-tied FPRM separate from transformers on binding and composition,
  consistent with Phase 2's findings.

The consolidated report will restate these on the natural-language format with
multi-seed confidence intervals.

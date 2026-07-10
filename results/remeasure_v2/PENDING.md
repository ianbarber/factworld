# Issue-#11 re-measure queue — pending jobs and fold instructions (2026-07-10)

Folded so far (branch `remeasure/v2-numbers`, nothing committed): E1b scaffolded leg on v2
(experiments §16, frontier report gap definition), dose-response v2 (§17, consolidated §4),
kimi long-context L256 (§18, consolidated §7), commutative calibration (§19, taxonomy rows +
frontier report Components), v2 reference baselines (§20, docs/results.md).

Also folded since: kimi L512 thinking — relaxed 0.96, Wilson [0.80, 0.99], empty 0.00, ~$1.21
(consolidated §7 table + experiments §18; record in
`results/composite_frontier_20260710.jsonl`).

Also folded since: ci_dissociation (GPU queue job 3, DONE 15:27Z rc=0, 76 min) — rebuilt
`docs/results-ci.md` verified cell-by-cell against `results/remeasure_v2/ci_dissociation.log`;
no report/README prose cites CI-dissociation numbers (grep `results-ci` hits only the README
file map). The doc's self-rendered header claim ("gdn is the binding specialist") contradicted
the rebuilt v2 tables (gdp ≥ gdn on binding at L4–L32; gdn seed-bimodal, stds to ±0.39) — the
header string in `scripts/ci_dissociation.py` and the rendered line were updated to match the
data.

## Still running / queued

### 1. curriculum_staged_v2 — the §5 flagship re-measure (GPU queue job 4, started 15:27Z, ~12–25 h, overnight)

- Master queue PID 1667899 (verified alive; runs jobs 4–5 in sequence); 3 archs × 3 seeds,
  d768×8, staged curriculum on v2 specs; output `results/curriculum_staged_v2_d768*`; log
  `results/remeasure_v2/curriculum_staged_v2.log`.
- Fold targets (all currently on retired `composite_copy_v1`):
  - `reports/factworld-consolidated.md` §5: the 0.747 ± 0.174 flagship table (gdp_hybrid /
    fprm / transformer), the relaxed-vs-last-N paragraph, and the per-leg decomposition table
    (0.969 / 0.747 / 0.747 etc.) — replace with v2 values, read p(converge) + per-leg
    decomposition, keep the compute-matched framing (10/76/101M params, FLOPs matched).
  - `reports/factworld-consolidated.md` §2 versioning note: drop "§5 local tables ... v1" once
    folded.
  - `reports/frontier-benchmark.md`: the two references to "the staged-curriculum recipe
    (consolidated §5, d768)" and the d768 decomposition numbers (binding 0.97 / value 0.75) in
    the breadth-sweep bullets and price table.
  - New experiments README section (house style: config, table, **Finding:**, Data:).
- Note: §5's "Scale robustness" subsection numbers come from composite_scale (next job), not
  this one — do not touch that subsection on this fold.

### 2. composite_scale_v2 — compute-matched scale sweep (GPU queue job 5, ~1–2 GPU-days)

- Log `results/remeasure_v2/composite_scale_v2.log`; output `results/composite_scale_*`.
- Fold: `reports/factworld-consolidated.md` §5 "Scale robustness (compute-matched sweep)"
  table (small/medium/large × 3 archs) and its bullets; README price-table rows that cite the
  breadth/scale evidence only if the ordering changes; experiments README section.

If the queue dies, relaunch the remaining `run_job` lines from
`scripts/gpu_queue_remeasure_v2.sh` (one at a time, same logs); ledger
`results/remeasure_v2/queue_status.log`.

## Deferred (with reasons)

- **MOPD binding domain**: owner-flagged confounded; large re-run; deferred with a note in
  issue #11 rather than re-measured.
- **Archived OpenRouter grids** (consolidated §4 grid, worked example): historical snapshot on
  v1, superseded by the frontier benchmark for cross-model claims — not re-run by design.
- **d256 binding sweep**: superseded by the 45-run breadth sweep (experiments §13, already v2).
- **Kimi long-context L64–L128/L1024**: discriminate-before-spending — glm's v2 curve is flat
  and kimi reads 1.00 @L256; buy further lengths only if a claim needs them.

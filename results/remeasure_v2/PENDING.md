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

## Adjudicated and folded (2026-07-10)

### 1. curriculum_staged_v2 — DONE (trace artifact adjudicated; trace-free rerun landed, folded 2026-07-11)

- The `--use_trace` launch (`results/curriculum_staged_v2_d768.jsonl`) stays excluded as a
  protocol artifact — relaxed structurally 0 on all nine runs, `contains` inflated; experiments
  §21 logs the adjudication.
- Trace-free rerun (identical command, no `--use_trace`) DONE:
  `results/curriculum_staged_v2_d768_notrace.{jsonl,md}`, log
  `results/remeasure_v2/curriculum_notrace.log`. composite_p16@L16 relaxed, 3 seeds/eval_n=500
  (mean±pstdev; pconv = seeds ≥0.9):
  - **gdp_hybrid 0.833±0.089** (0.758/0.782/0.958; pconv 1/3; holder ≥0.998, value 0.833;
    contains ≈ relaxed, last-N 0.00 — no artifact signature)
  - fprm 0.109±0.089 (0.056/0.036/0.234; binding ≥0.994, value leg collapsed)
  - transformer 0.001±0.001 (floor on both legs)
- This is the §5 flagship cell; the scale sweep's 2-seed/eval_n=200 medium cell (0.732±0.013)
  stays as corroboration in the scale table. Folded: consolidated §2 versioning note + §5
  (flagship table, v1-comparison paragraph, relaxed-vs-last-N, per-leg decomposition,
  scale-table framing), frontier report local-regime lines, experiments §23 (+ §21/§22
  pointers).

Nothing remains pending in this queue; the only open item repo-wide is the deferred MOPD
re-pin below.

### 2. composite_scale_v2 — DONE, folded

- `results/composite_scale_20260710_221530.jsonl` (recomputed from raw, matches the md).
  VALID: trace-free, v2 staged specs. Its medium cell (the exact §5 recipe) corroborates the
  3-seed flagship (item 1): gdp_hybrid composite_p16@L16 relaxed 0.732±0.013 (0.720/0.745, holder 1.00 both
  seeds); fprm 0.033±0.012; transformer 0.005±0.005. Small gdp fails the value leg (0.12±0.08,
  holder 1.0 — v1's 0.98 was sampler-flattered); large gdp seed-bimodal with a genuine
  (contains 0.000) value-leg failure.
- Folded: consolidated §5 (flagship table, relaxed-vs-last-N, decomposition, scale-robustness
  table + bullets, §2 versioning note), frontier report local-regime lines (breadth-sweep
  bullet, d768 decomposition citation, re-measure paragraph, price-table close), experiments
  §22.

If relaunching GPU work, use the `run_job` lines from `scripts/gpu_queue_remeasure_v2.sh`
(minus `--use_trace` for job 4); ledger `results/remeasure_v2/queue_status.log`.

## Deferred (with reasons)

- **MOPD binding domain**: owner-flagged confounded; large re-run; deferred with a note in
  issue #11 rather than re-measured.
- **Archived OpenRouter grids** (consolidated §4 grid, worked example): historical snapshot on
  v1, superseded by the frontier benchmark for cross-model claims — not re-run by design.
- **d256 binding sweep**: superseded by the 45-run breadth sweep (experiments §13, already v2).
- **Kimi long-context L64–L128/L1024**: discriminate-before-spending — glm's v2 curve is flat
  and kimi reads 1.00 @L256; buy further lengths only if a claim needs them.

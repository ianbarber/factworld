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

### 1. curriculum_staged_v2 — NEEDS-RERUN (protocol artifact; runs excluded, not folded)

- Ran to completion (`results/curriculum_staged_v2_d768.jsonl`) but was launched with
  `--use_trace` (`scripts/gpu_queue_remeasure_v2.sh` job 4); the v1 flagship it re-measures was
  `use_trace=False`. Trace-first emission makes the prefix-committed relaxed metric
  structurally 0 (all nine runs 0.000) while `contains` inflates (gdp p5 0.981) — the known
  artifact signature; adjudicated on the raw records as artifact, reproducing the v1 trace-mode
  control (`results/curriculum_staged_d768_b64_80k_trace.md`, composite 0.00). Composite
  capability is unmeasurable under this protocol; no per-example preds or checkpoints stored,
  so no rescoring. Experiments §21 logs the artifact.
- Corrected rerun (identical command, NO `--use_trace`; ~12–25 GPU-h):

  ```
  python scripts/experiment_curriculum_staged.py \
      --archs gdp_hybrid,fprm,transformer --seeds 0 1 2 \
      --d_model 768 --n_layers 8 --batch 128 --train_n 80000 --eval_n 500 \
      --schedule 'binding:0.5,recall_easy:0.5:10000;binding:0.25,recall_med:0.35,composite_p5:0.4:7500;binding:0.15,recall_hard:0.25,composite_p5:0.25,composite_p16:0.35:7500' \
      --out_prefix results/curriculum_staged_v2_d768_notrace
  ```

- On landing, it upgrades the §5 flagship cell from 2 seeds/eval_n=200 (scale-sweep medium) to
  3 seeds/eval_n=500; the fold targets are already on v2, so only the numbers move.

### 2. composite_scale_v2 — DONE, folded

- `results/composite_scale_20260710_221530.jsonl` (recomputed from raw, matches the md).
  VALID: trace-free, v2 staged specs. Its medium cell (the exact §5 recipe) is the standing v2
  flagship: gdp_hybrid composite_p16@L16 relaxed 0.732±0.013 (0.720/0.745, holder 1.00 both
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

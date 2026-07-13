# MOPD v2 re-pin — DONE (issue #11, last open item; folded 2026-07-12)

Working note for the run, kept as the run record. Branch `mopd/v2-repin`. Pipeline launched
2026-07-12 16:08, **COMPLETE 19:54** (`pipeline.log`; ~3.8 h wall on the RTX 5090). All stage
gates passed; fold completed the same day (see "Fold — completed" below).

## What changed (the re-pin)

Binding domain re-pinned from the retired `binding_v1` (recency-defective sampler) to
`CANONICAL["binding_v2"]` (uniform resolving-write position), same knobs otherwise:

- `experiments/mopd/mopd_hf.py` `DOMAINS["binding"]` → `TK.CANONICAL["binding_v2"].scaled(train_lengths=(8,16), eval_lengths=(16,24,32))`
- `experiments/mopd/mopd.py` `binding_spec()` → `_shared(TK.CANONICAL["binding_v2"])` (from-scratch control path)
- `experiments/mopd/bench_qwen.py` `BENCH` → `binding_v2` and `composite_copy_v2` (CANONICAL names only)

Recall domain (`recall_copy_v1`, CANONICAL) unchanged. `tests/test_mopd.py` passes on the
re-pinned specs (tokenizer covers v2; losses finite; ckpt round-trip).

Verified before launch: 10 generated `binding_v2` items recompute gold from the rendered
prompt (last give-event for the queried object == answer, 10/10); `reward()` (canonical
relaxed match) scores synthetic correct/incorrect/formatted answers per the documented
prefix-of-answer-span semantics.

## Baseline (2026-07-12): the base sits AT the object-filter floor on v2

Qwen3-1.7B base, greedy, no thinking, relaxed match, n=200 (floors computed on the exact
deterministic test items, n=300; chance = 1/k = 0.200; floors independently recomputed at
fold — they reproduce exactly):

| L | base binding_v2 | object-filter floor E[1/w] | recency heuristic | chance |
|---|---|---|---|---|
| 16 | 0.420 | 0.471 | 0.177 | 0.200 |
| 24 | 0.325 | 0.348 | 0.220 | 0.200 |
| 32 | 0.290 | 0.261 | 0.160 | 0.200 |

**Plain reading: the base drops to the honest shallow bound.** At every eval length the
base sits at (or just under) the object-filter floor and decays with the floor's ~1/L
shape — consistent with filtering events by the queried object and guessing among its
recipients, with no last-write resolution above that. The published "partial capability"
premise (0.325–0.39 on v1, at v1's recency-heuristic level) does not survive v2: on the
clean sampler the base has **no measurable binding capability above the shallow floor**.

Base recall (unchanged domain) reproduces: L16=0.390, L24=0.245 (published 0.34/0.28,
same band; different GPU/transformers stack). Full bench (`bench_qwen.md`, n=40):
recall_copy@6 relaxed 0.675 / contains 0.925; conflict@4 0.475 / 0.90; **binding_v2@16
0.375**; composite_copy_v2@16 0.000; chain@4 0.075; s5@32 0.000 / 0.125.

## Run — complete (all stages DONE, `pipeline.log`)

Driver `run_pipeline.sh`, `.venv-train/bin/python` (peft 0.19.1 + accelerate 1.14.0),
GPU-serialized on the RTX 5090:

| stage | command | output | duration |
|---|---|---|---|
| stage2_teachers | `hf_teachers.py --steps 300` | `hf_teachers.md`, adapters `ckpts_hf/teacher_*` | 15 min |
| stage3_mopd_pg | `hf_mopd.py --loss pg` | `hf_mopd.md`, `ckpts_hf/student_pg` | 30 min |
| stage3_mopd_kl | `hf_mopd.py --loss kl` | `hf_mopd.md`, `ckpts_hf/student_kl` | 30 min |
| stage4_evaluate | `hf_evaluate.py` | `hf_evaluate.md` (single-seed table) | 2 min |
| stage5_seeds | `hf_seeds.py --seeds 0 1 2` | `hf_seeds.md` (**headline** mean±std) | 2.5 h |

## Fold — completed (2026-07-12)

Stage gates, adjudicated from the raw logs (every table cross-checked against its stage log;
normalised scores and seed mean±std recomputed independently — all reproduce):

1. **stage2_teachers — the key gate PASSED.** Teacher binding 0.975/0.980/0.910 at L16/24/32
   vs floors 0.471/0.348/0.261 (base 0.420/0.325/0.290): outcome-RL lifts from the shallow
   floor to near-solved, including the held-out L24/L32 where the floor keeps falling. The
   stronger reframing ("floor → genuine last-write tracking") is supported and published.
2. **stage3/4 — sane.** Distillation reverse-KL collapses to ≈0 within ~40 steps and stays
   there; single-seed normalised avg mopd_pg 0.987 / mopd_kl 0.997. One calibration note
   (logged in experiments §28): the binding KL starts materially nonzero (pg 6.4, kl 1.6 at
   the first log point) because the v2 binding teacher moves far from the base — stability
   holds regardless.
3. **stage5_seeds — headline landed.** mopd_pg binding 1.051±0.031 / recall 1.002±0.003
   (avg 1.027±0.033); mopd_kl 1.083±0.043 / 1.000±0.004 (avg 1.042±0.052); every seed ≥1 on
   both domains, both loss forms. teacher_recall stays at the binding floor (0.094±0.021);
   teacher_binding→recall transfer 0.889±0.086.

Docs updated per the must-change list: `experiments/mopd/README.md` (contained sampler note
replacing the confound caveat; floor-referenced abstract, §2 bench, §3 tables with floor rows,
§5 re-checked limitation — the student-exceeds-teacher effect holds vs 150-step seed teachers,
flattens to parity vs 300-step teachers; the v1-measured thinking aside dropped),
`experiments/mopd/REPRODUCE.md` (v2 expected headlines), from-scratch control tables
(`stage1_base.md`, `probe_headroom.md`, `stage2_teachers.md`) annotated as v1-sampler-era
provenance rather than re-run, `docs/experiments/README.md` §28 (the log entry), and issue #11
commented with the final numbers. Unaffected sections untouched (recall domain, chain-RL wall,
MOPD mechanism claims, from-scratch conclusion).

# MOPD on FactWorld — composing two RL-specialised abilities into one model

*Ian Barber — July 2026*

A test of **MOPD (Multi-teacher On-Policy Distillation)** — [arXiv:2606.30406](https://arxiv.org/abs/2606.30406),
Xiaomi/PKU — on the FactWorld instrument. MOPD's recipe: take a shared base, RL-specialise it
independently per domain to get a set of teachers, then distil the teachers back into a single
student *on the student's own rollouts* (routing each rollout to its domain teacher and matching
its per-token distribution). The student ends up holding every teacher's ability at once.

FactWorld is a good testbed: every task has a symbolic-oracle gold answer, so RL gets a clean
**verifiable reward**, and the tasks are small enough to run the full pipeline on one consumer
GPU (validated on an RTX 3090; the v2 numbers below took ~3.8 h on an RTX 5090).

> **Sampler note (issue [#11](https://github.com/ianbarber/factworld/issues/11), resolved).**
> An earlier run of this pipeline used the retired `binding_v1` sampler, whose resolving write
> clusters near the stream end; the base's apparent "partial capability" there (0.33) was that
> sampler's recency artifact. All numbers below are from the re-pinned CANONICAL `binding_v2`
> (uniform resolving-write position), on which the base sits at the documented object-filter
> floor E[1/w] — so the normalised score's 0-anchor is an honest shallow bound and the RL lift
> is read against the floor, not against zero.

## Abstract

We RL-specialise a pretrained **Qwen3-1.7B** on two distinct FactWorld abilities — **binding**
(last-write-wins state tracking) and **recall** (associative retrieval under a large distractor
pool) — then MOPD-distil both specialists into one model. Every model is a LoRA adapter on one
frozen Qwen3 backbone, so the same-origin condition the paper shows is load-bearing is *literal*
(initial student↔teacher KL ≈ 0). The base starts AT the shallow bound on binding (0.42 @L16 vs
the 0.47 object-filter floor) and partial on recall (0.39); outcome-RL lifts both to near-solved
(binding 0.98 @L16 and 0.91 at the held-out L32, recall 0.99), and across 3 seeds a **single
distilled adapter matches or exceeds both specialist teachers on both domains** (normalised
score 1.03±0.03 avg for the policy-gradient form, 1.04±0.05 for exact reverse-KL), whereas no
single teacher reliably does both. A from-scratch control shows why a pretrained base matters:
outcome-RL barely specialises a small from-scratch transformer on these
tasks (they are supervision-density-limited), and `chain`/`s5` stay at floor under outcome-RL.

## 1. Method

One **frozen Qwen3-1.7B** backbone carries several **LoRA adapters** (`peft`, `r=16`,
`all-linear`):

- **base** — adapters disabled: the pretrained backbone (normalised-score 0-anchor).
- **teacher_binding, teacher_recall** — one GRPO-trained adapter per domain (1-anchors).
- **student_pg, student_kl** — the MOPD-distilled adapters (two loss forms).

Because every adapter shares identical backbone weights, all models are same-origin by
construction and the initial student↔teacher KL is ≈0 (LoRA zero-init). Distillation swaps in the
teacher adapter for a no-grad scoring pass, then updates the student adapter; only adapters train,
so base + all teachers + student fit in ~6 GB.

**Stage 2 (teachers).** GRPO with the verifiable relaxed-match reward on the answer span
(`factworld.tasks.score_relaxed` after `Renderer.normalize`), thinking disabled so rollouts are a
clean 1–2-token span. Group-normalised advantage; fresh (non-repeating) prompts each step.

**Stage 3 (MOPD).** Student rolls out on-policy (N=1); each rollout is routed to its domain
teacher, which scores it; the student is updated on the per-token **reverse KL** toward the teacher.
Two loss forms (both from the paper): `pg` = clipped teacher−student log-diff advantage (eq. 4);
`kl` = **exact full-vocabulary** reverse KL (affordable because answers are short).

**Metric.** Normalised score `s̃ = (model − base)/(teacher − base)` — 0 at the base backbone, 1 at
the per-domain teacher — averaged over eval lengths and domains.

## 2. Choosing the domains (bench)

`bench_qwen.py` — Qwen3-1.7B on FactWorld, no thinking, greedy, relaxed match:

| task | acc | verdict |
|---|---|---|
| recall_copy@6 | 0.93 (contains) | near-ceiling |
| conflict@4 | 0.90 (contains) | near-ceiling |
| **binding_v2@16** | **0.375** | **at the object-filter floor (E[1/w] = 0.47) → the sharp RL question** |
| composite_copy_v2@16 | 0.00 | floored (emits 1 token, needs 2) |
| chain@4 | 0.08 | floored |
| s5@32 | 0.00 | floored |

We pick two domains with real headroom and distinct computations. **Binding**: the base sits at
the object-filter floor at every eval length (0.42/0.33/0.29 vs E[1/w] 0.47/0.35/0.26 at
L16/24/32, decaying with the floor's ~1/L shape) — consistent with filtering events by the
queried object and guessing among its recipients, with no last-write resolution above that. So
outcome-RL is asked the sharp question: lift OFF the shallow bound to genuine last-write
tracking, not improve an already-partial ability. **Recall under a large distractor pool**
(`recall_copy` scaled to pool≈16, where the base drops to ≈0.25–0.4) is genuinely partial.
Thinking raises `contains` accuracy but breaks the clean answer span and makes rollouts ~20×
longer, so we keep it off.

## 3. Results

**Stage 2 — outcome-RL lifts the base from the shallow floor to near-solved** (`hf_teachers.md`,
GRPO LoRA, greedy n=200; floor = object-filter E[1/w] on the same deterministic test items;
binding trains at L∈{8,16}, so L24/L32 are held-out lengths):

| domain | floor | base | teacher | Δ |
|---|---|---|---|---|
| binding L16 | 0.47 | 0.42 | 0.98 | +0.56 |
| binding L24 (held-out) | 0.35 | 0.33 | 0.98 | +0.66 |
| binding L32 (held-out) | 0.26 | 0.29 | 0.91 | +0.62 |
| recall L16 | — | 0.39 | 0.99 | +0.60 |
| recall L24 (held-out) | — | 0.25 | 1.00 | +0.75 |

Reward climbs cleanly (binding 0.52→0.97 by step 100, ~1.00 from step 200; recall 0.92→1.00 in
~60 steps), and the lift is not a longer-range shallow heuristic: at the held-out lengths the
floor keeps falling (~1/L) while the teacher holds 0.91–0.98 — the teacher is resolving the last
write, which is exactly what the base could not do above the floor.

**Stage 3 — one student holds BOTH abilities**, across 3 seeds (`hf_seeds.md`, normalised score,
mean±std over seeds 0/1/2; each seed = fresh teachers + fresh students):

| model | binding | recall | avg |
|---|---|---|---|
| base | 0.00 | 0.00 | 0.00 |
| teacher_binding | 1.00±0.00 | 0.89±0.09 | 0.94±0.08 |
| teacher_recall | 0.09±0.02 | 1.00±0.00 | 0.55±0.45 |
| **mopd_pg** | **1.05±0.03** | **1.00±0.00** | **1.03±0.03** |
| **mopd_kl** | **1.08±0.04** | **1.00±0.00** | **1.04±0.05** |

Every seed, the **single MOPD adapter matches or exceeds both teachers on both domains** (binding
0.42→0.98, recall 0.39→0.99 raw @L16; recall norm is rock-steady at 1.00±0.00, binding ranges
1.01–1.14 and always beats the binding teacher because it extrapolates better to the held-out
L24/L32). The two loss forms are comparable (PG 1.03, KL 1.04), reproducing the paper's finding
that policy-gradient and reverse-KL distillation perform alike under same-origin teachers.
Distillation is stable: the measured per-token reverse KL collapses to ≈0 within ~40 steps and
stays there (`hf_mopd.md`) — on binding it starts materially nonzero (the binding teacher has
moved far from the base on the clean sampler) and still converges without drama; on recall it
starts ≈0.

One honest wrinkle the seeds expose: `teacher_binding` also **transfers to recall** (0.89±0.09) —
GRPO on binding incidentally teaches the clean single-token output format, and recall is
format-limited on the base (bench `contains` 0.93 vs `relaxed` 0.68 at the easy pool; 0.39 on
the scaled domain). But that transfer is below MOPD and seed-variable (0.79–1.00), and
`teacher_recall` still cannot do binding (0.09±0.02 — raw scores at or barely above the
object-filter floor), so *no single teacher reliably does both* while the MOPD student does —
every seed.

## 4. Why a pretrained base — the from-scratch control

The original plan RL-specialised a small **from-scratch** transformer (the repo's own model zoo).
On FactWorld's tasks that barely works: outcome-RL lifts a from-scratch binding/recall model only
marginally and non-monotonically, and `chain` does not move under outcome-RL — the base sits at the random
floor even in-distribution and GRPO merely memorises the training pool (0 generalisation),
matching the archived s5-GRPO negative result. These tasks are **supervision-density-limited** (the
FactWorld thesis: dense SFT climbs them, sparse outcome-reward does not), so a from-scratch base
gives MOPD little to distil. A pretrained base already has the circuits; RL only has to *select*
them, which it does cleanly. The from-scratch path is preserved as a control
(`stage1_base.py`, `probe_headroom.py`, `stage2_teachers.py`; tables in `*_base.md` /
`stage2_teachers.md`; CPU infra tests in `tests/test_mopd.py`). The control's tables are
v1-sampler-era runs (`mopd.py` now pins `binding_v2`); the conclusion they support is
qualitative and sampler-independent.

## 5. Limitations

- One base (Qwen3-1.7B), two domains, 3 seeds — a demonstration, not a scaling study.
- Reward = relaxed match on a short span with thinking off; the harder reasoning-gated tasks
  (`composite`, `s5`) are out of scope here (they need thinking / dense supervision).
- The student *exceeds* the teachers on binding against the 150-step seed teachers (norm >1
  every seed, 1.01–1.14, both loss forms): it extrapolates to the held-out L24/L32 better than
  the binding teacher. Against the longer-trained 300-step single-seed teachers the students
  match rather than exceed (0.97/0.99, `hf_evaluate.md`) — a real but teacher-strength-dependent
  multi-teacher effect.

## Reproduce

See [`REPRODUCE.md`](REPRODUCE.md). Single seed: `bench_qwen.py` → `hf_teachers.py` → `hf_mopd.py
--loss pg` / `--loss kl` → `hf_evaluate.py`. Multi-seed robustness (the headline table):
`hf_seeds.py --seeds 0 1 2`. Needs a CUDA GPU + `transformers`, `peft`, `accelerate`; the
data/oracle/eval layer is pure-stdlib.

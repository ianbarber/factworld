# MOPD on FactWorld — composing two RL-specialised abilities into one model

*Ian Barber — July 2026*

A test of **MOPD (Multi-teacher On-Policy Distillation)** — [arXiv:2606.30406](https://arxiv.org/abs/2606.30406),
Xiaomi/PKU — on the FactWorld instrument. MOPD's recipe: take a shared base, RL-specialise it
independently per domain to get a set of teachers, then distil the teachers back into a single
student *on the student's own rollouts* (routing each rollout to its domain teacher and matching
its per-token distribution). The student ends up holding every teacher's ability at once.

FactWorld is a good testbed: every task has a symbolic-oracle gold answer, so RL gets a clean
**verifiable reward**, and the tasks are small enough to run the full pipeline on one RTX 3090.

## Abstract

We RL-specialise a pretrained **Qwen3-1.7B** on two distinct FactWorld abilities — **binding**
(last-write-wins state tracking) and **recall** (associative retrieval under a large distractor
pool) — then MOPD-distil both specialists into one model. Every model is a LoRA adapter on one
frozen Qwen3 backbone, so the same-origin condition the paper shows is load-bearing is *literal*
(initial student↔teacher KL ≈ 0). Outcome-RL lifts each domain sharply (binding 0.34→0.99, recall
0.34→1.00), and a **single distilled adapter matches or exceeds both specialist teachers on both
domains** (normalised score 1.05 avg for the policy-gradient form, 1.03 for exact reverse-KL),
whereas each teacher is strong only on its own domain. A from-scratch control shows why a
pretrained base matters: outcome-RL barely specialises a small from-scratch transformer on these
tasks (they are supervision-density-limited), and `chain`/`s5` are outright RL walls.

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
| recall_copy@6 | 0.95 (contains) | near-ceiling |
| conflict@4 | 0.90 (contains) | near-ceiling |
| **binding@16** | **0.33** | **partial → RL headroom** |
| composite_copy@16 | 0.00 | wall (emits 1 token, needs 2) |
| chain@4 | 0.08 | wall |
| s5@32 | 0.00 | wall |

We pick the two domains where the base is genuinely *partial* (real headroom, distinct
computations): **binding** and **recall under a large distractor pool** (`recall_copy` scaled to
pool≈16, where the base drops to ≈0.3–0.5). Thinking raises `contains` accuracy sharply (binding
0.33→0.94) but breaks the clean answer span and makes rollouts ~20× longer, so we keep it off.

## 3. Results

**Stage 2 — RL genuinely specialises the pretrained base** (`hf_teachers.md`, GRPO LoRA):

| domain | base | teacher | Δ |
|---|---|---|---|
| binding L16 | 0.34 | 0.99 | +0.65 |
| binding L32 (untrained length) | 0.28 | 0.89 | +0.61 |
| recall L16 | 0.34 | 0.99 | +0.65 |
| recall L24 | 0.28 | 1.00 | +0.72 |

Reward climbs cleanly and monotonically (binding 0.55→0.99, recall 0.93→1.00 in ~150 steps), and
the binding teacher even extrapolates to lengths it was not trained on.

**Stage 3 — one student holds BOTH abilities** (`hf_evaluate.md`, n=500, normalised score):

| model | binding | recall | avg |
|---|---|---|---|
| base | 0.00 | 0.00 | 0.00 |
| teacher_binding | **1.00** | 0.25 | 0.62 |
| teacher_recall | 0.03 | **1.00** | 0.52 |
| **mopd_pg** | **1.09** | **1.01** | **1.05** |
| **mopd_kl** | **1.05** | **1.01** | **1.03** |

Each teacher is a specialist — strong only on its own domain — but the **single MOPD adapter
matches or slightly exceeds both on both domains** from one set of weights (binding 0.34→0.99,
recall 0.34→1.00 in raw accuracy). The two loss forms are comparable (PG 1.05, KL 1.03),
reproducing the paper's finding that policy-gradient and reverse-KL distillation perform alike
under same-origin teachers. Distillation is stable: the per-token reverse KL starts low and
converges (`hf_mopd.md`), the expected consequence of the ≈0 initial same-origin KL.

## 4. Why a pretrained base — the from-scratch control

The original plan RL-specialised a small **from-scratch** transformer (the repo's own model zoo).
On FactWorld's tasks that barely works: outcome-RL lifts a from-scratch binding/recall model only
marginally and non-monotonically, and `chain` is an outright RL wall — the base sits at the random
floor even in-distribution and GRPO merely memorises the training pool (0 generalisation),
matching the archived s5-GRPO negative result. These tasks are **supervision-density-limited** (the
FactWorld thesis: dense SFT climbs them, sparse outcome-reward does not), so a from-scratch base
gives MOPD little to distil. A pretrained base already has the circuits; RL only has to *select*
them, which it does cleanly. The from-scratch path is preserved as a control
(`stage1_base.py`, `probe_headroom.py`, `stage2_teachers.py`; tables in `*_base.md` /
`stage2_teachers.md`; CPU infra tests in `tests/test_mopd.py`).

## 5. Limitations

- One base (Qwen3-1.7B), two domains, single seed — a demonstration, not a scaling study.
- Reward = relaxed match on a short span with thinking off; the harder reasoning-gated tasks
  (`composite`, `s5`) are out of scope here (they need thinking / dense supervision).
- The student slightly *exceeds* the teachers (avg >1): a small multi-teacher synergy, but also
  within eval noise at these lengths — read it as "fully recovers both," not "reliably beats."

## Reproduce

See [`REPRODUCE.md`](REPRODUCE.md). Order: `bench_qwen.py` → `hf_teachers.py` → `hf_mopd.py
--loss pg` / `--loss kl` → `hf_evaluate.py`. Needs a CUDA GPU + `transformers`, `peft`,
`accelerate`; the data/oracle/eval layer is pure-stdlib.

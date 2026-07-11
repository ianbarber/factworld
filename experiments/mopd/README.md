# MOPD on FactWorld — composing two RL-specialised abilities into one model

*Ian Barber — July 2026*

A test of **MOPD (Multi-teacher On-Policy Distillation)** — [arXiv:2606.30406](https://arxiv.org/abs/2606.30406),
Xiaomi/PKU — on the FactWorld instrument. MOPD's recipe: take a shared base, RL-specialise it
independently per domain to get a set of teachers, then distil the teachers back into a single
student *on the student's own rollouts* (routing each rollout to its domain teacher and matching
its per-token distribution). The student ends up holding every teacher's ability at once.

FactWorld is a good testbed: every task has a symbolic-oracle gold answer, so RL gets a clean
**verifiable reward**, and the tasks are small enough to run the full pipeline on one RTX 3090.

> **Confound caveat (issue [#11](https://github.com/ianbarber/factworld/issues/11)).** The
> binding domain here was trained and evaluated on the retired `binding_v1` sampler, whose
> queried object's resolving write clusters near the stream end. The base model's "partial
> capability" (0.33/0.34) sits at that sampler's recency-heuristic floor (~0.4), so the +0.65
> outcome-RL lift is confounded with learning the recency shortcut, not necessarily
> last-write-wins tracking. The MOPD *mechanism* result (two teachers distil into one student)
> is unaffected, but the binding numbers should not be read as a state-tracking capability
> claim. Re-measure on `binding_v2` pending (~0.5–1 GPU-day; pins to update: the domain specs
> in `mopd.py`, `mopd_hf.py` and `bench_qwen.py`).

## Abstract

We RL-specialise a pretrained **Qwen3-1.7B** on two distinct FactWorld abilities — **binding**
(last-write-wins state tracking) and **recall** (associative retrieval under a large distractor
pool) — then MOPD-distil both specialists into one model. Every model is a LoRA adapter on one
frozen Qwen3 backbone, so the same-origin condition the paper shows is load-bearing is *literal*
(initial student↔teacher KL ≈ 0). Outcome-RL lifts each domain sharply (binding 0.34→0.99, recall
0.34→1.00), and across 3 seeds a **single distilled adapter matches or exceeds both specialist
teachers on both domains** (normalised score 1.07±0.12 avg for the policy-gradient form,
1.06±0.09 for exact reverse-KL), whereas no single teacher reliably does both. A from-scratch
control shows why a
pretrained base matters: outcome-RL barely specialises a small from-scratch transformer on these
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
| recall_copy@6 | 0.95 (contains) | near-ceiling |
| conflict@4 | 0.90 (contains) | near-ceiling |
| **binding@16** | **0.33** | **partial → RL headroom** |
| composite_copy@16 | 0.00 | floored (emits 1 token, needs 2) |
| chain@4 | 0.08 | floored |
| s5@32 | 0.00 | floored |

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

**Stage 3 — one student holds BOTH abilities**, across 3 seeds (`hf_seeds.md`, normalised score,
mean±std over seeds 0/1/2; each seed = fresh teachers + fresh students):

| model | binding | recall | avg |
|---|---|---|---|
| base | 0.00 | 0.00 | 0.00 |
| teacher_binding | 1.00±0.00 | 0.85±0.15 | 0.92±0.13 |
| teacher_recall | 0.03±0.03 | 1.00±0.00 | 0.51±0.49 |
| **mopd_pg** | **1.14±0.14** | **1.00±0.02** | **1.07±0.12** |
| **mopd_kl** | **1.11±0.11** | **1.01±0.01** | **1.06±0.09** |

Every seed, the **single MOPD adapter matches or exceeds both teachers on both domains** (binding
0.34→0.99, recall 0.34→1.00 raw; recall norm is rock-steady at 1.00±0.02, binding ranges 1.02–1.33
and always beats the binding teacher because it extrapolates better to L24/L32). The two loss forms
are comparable (PG 1.07, KL 1.06), reproducing the paper's finding that policy-gradient and
reverse-KL distillation perform alike under same-origin teachers. Distillation is stable: the
per-token reverse KL starts low and converges (`hf_mopd.md`), the expected consequence of the ≈0
initial same-origin KL.

One honest wrinkle the seeds expose: `teacher_binding` also **transfers to recall** (0.85±0.15) —
GRPO on binding incidentally teaches the clean single-token output format, and recall was
format-limited on the base (`contains`=0.95 but `relaxed`=0.34). But that transfer is unstable and
below MOPD, and `teacher_recall` still cannot do binding (0.03), so *no single teacher reliably
does both* while the MOPD student does — every seed.

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
`stage2_teachers.md`; CPU infra tests in `tests/test_mopd.py`).

## 5. Limitations

- One base (Qwen3-1.7B), two domains, 3 seeds — a demonstration, not a scaling study.
- Reward = relaxed match on a short span with thinking off; the harder reasoning-gated tasks
  (`composite`, `s5`) are out of scope here (they need thinking / dense supervision).
- The student *exceeds* the teachers on binding (norm >1 every seed): it extrapolates to longer
  give-streams better than the binding teacher — a small but consistent multi-teacher effect rather
  than a within-noise blip (recall std is ±0.01–0.02).

## Reproduce

See [`REPRODUCE.md`](REPRODUCE.md). Single seed: `bench_qwen.py` → `hf_teachers.py` → `hf_mopd.py
--loss pg` / `--loss kl` → `hf_evaluate.py`. Multi-seed robustness (the headline table):
`hf_seeds.py --seeds 0 1 2`. Needs a CUDA GPU + `transformers`, `peft`, `accelerate`; the
data/oracle/eval layer is pure-stdlib.

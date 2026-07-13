# Reproducing the FactWorld MOPD study

Reproduction guide for **[`README.md`](README.md)**. This is a **segregated** experiment under
`experiments/mopd/` — it adds code and reuses the repo's frozen `factworld/` instrument; it
changes nothing in `reports/` / `docs/`. Every stage writes a `<name>.md` table next to it on
completion (crash-safe). Checkpoints/adapters go to gitignored dirs
(`ckpts/` for the from-scratch path, `ckpts_hf/` for the Qwen path).

Validated on a single RTX 3090 (24 GB); the published v2 numbers come from an RTX 5090, where
the full pipeline (stages 2–5 below, `results/mopd_v2/run_pipeline.sh`) takes ~3.8 h wall. The
Qwen path needs `transformers`, `peft`, `accelerate`
(`uv pip install --python .venv/bin/python peft accelerate`); the from-scratch path needs the
`train` extra (torch + flash-linear-attention).

## Primary result — MOPD on pretrained Qwen3-1.7B

A frozen Qwen3-1.7B backbone carries LoRA adapters: **base** = adapters off; **teacher_\*** =
one GRPO-trained adapter per domain; **student** = the MOPD-distilled adapter. Same-origin is
literal (identical backbone). Run in order:

| Claim (README §) | Script | Expected headline |
|---|---|---|
| Base at the object-filter floor on binding, partial on recall, at ceiling/floor elsewhere (pick domains) | `bench_qwen.py --n 40` | binding_v2≈0.38 (E[1/w] floor 0.47 @L16), recall(easy) contains≈0.93, composite/chain/s5≈0 → `bench_qwen.md` |
| Outcome-RL lifts the base from the shallow floor to near-solved | `hf_teachers.py --steps 300` | binding 0.42→0.98 @L16 (floor 0.47), 0.29→0.91 @held-out L32 (floor 0.26); recall 0.39→0.99 → `hf_teachers.md` |
| MOPD distils both teachers into one student (PG form) | `hf_mopd.py --loss pg` | reverse-KL converges to ≈0 within ~40 steps (binding starts nonzero); adapter saved → `hf_mopd.md` |
| Same with exact reverse-KL form | `hf_mopd.py --loss kl` | comparable dynamics → `hf_mopd.md` |
| One student holds BOTH abilities (single seed) | `hf_evaluate.py` | MOPD avg normalised score ≈ 1 on both domains (pg 0.99, kl 1.00) → `hf_evaluate.md` |
| **Robust across seeds (headline)** | `hf_seeds.py --seeds 0 1 2` | MOPD ≥1 on both domains every seed; mopd_pg 1.03±0.03, mopd_kl 1.04±0.05 avg → `hf_seeds.md` |

Infra smoke (loads the model; needs GPU): `python experiments/mopd/mopd_hf.py`.

## Motivation / negative control — from-scratch RL is too weak

Why a pretrained base: on FactWorld's tasks, outcome-RL barely specialises a **from-scratch**
tiny transformer (these tasks are supervision-density-limited — the repo's own thesis; cf. the
archived s5-GRPO negative result). That is what motivated the pivot. This path is preserved:

| Claim | Script | Expected headline |
|---|---|---|
| Infra correctness (losses, routing, ckpt round-trip) — CPU-ok | `python tests/test_mopd.py` | all smoke checks PASS |
| A weak shared base with RL headroom on both domains | `stage1_base.py --steps 15000` | binding/recall ≈0.4, pass@8≫greedy → `stage1_base.md`, `probe_headroom.md` |
| Chain **does not move under outcome-RL**; binding/recall lift only marginally from scratch | `stage2_teachers.py --lr 3e-4` | teacher−base small/non-monotone → `stage2_teachers.md` |

The committed control tables are v1-sampler-era runs; the scripts now pin `binding_v2`, so a
re-run reproduces the qualitative conclusion (small/non-monotone lift), not the exact cells.

## Notes

- **Same-origin invariant.** All Qwen adapters share one frozen backbone → initial
  student↔teacher KL ≈ 0 (LoRA zero-init), the stability condition the paper shows is
  load-bearing. `mopd_hf.py` self-check prints this KL (≈0).
- **No thinking.** Qwen rollouts run with `enable_thinking=False`: answers are a clean 1–2-token
  span (fast RL, verifiable reward). Thinking raises `contains` accuracy but breaks the clean
  span and makes rollouts ~20× longer (see `bench_qwen.py --think 1`).
- **Domains** and difficulty live in `mopd_hf.DOMAINS` (binding = last-write-wins state, on
  `binding_v2` — the base sits at the object-filter floor E[1/w]; recall = associative retrieval
  under a large distractor pool — partial on the base).

# Reproducing the FactWorld MOPD study

Reproduction guide for **[`README.md`](README.md)**. This is a **segregated** experiment under
`experiments/mopd/` — it adds code and reuses the repo's frozen `factworld/` instrument; it
changes nothing in `reports/` / `docs/`. Every stage writes a `<name>.md` table next to it on
completion (crash-safe). Checkpoints/adapters go to gitignored dirs
(`ckpts/` for the from-scratch path, `ckpts_hf/` for the Qwen path).

Validated on a single RTX 3090 (24 GB). The Qwen path needs `transformers`, `peft`, `accelerate`
(`uv pip install --python .venv/bin/python peft accelerate`); the from-scratch path needs the
`train` extra (torch + flash-linear-attention).

## Primary result — MOPD on pretrained Qwen3-1.7B

A frozen Qwen3-1.7B backbone carries LoRA adapters: **base** = adapters off; **teacher_\*** =
one GRPO-trained adapter per domain; **student** = the MOPD-distilled adapter. Same-origin is
literal (identical backbone). Run in order:

| Claim (README §) | Script | Expected headline |
|---|---|---|
| Base is PARTIAL on binding/recall, at ceiling/floor elsewhere (pick domains) | `bench_qwen.py --n 40` | binding≈0.33, recall(easy)≈0.95, composite/chain/s5≈0 → `bench_qwen.md` |
| Outcome-RL genuinely specialises the pretrained base | `hf_teachers.py --steps 300` | teacher ≫ base per domain (binding +~0.6, recall→1.0) → `hf_teachers.md` |
| MOPD distils both teachers into one student (PG form) | `hf_mopd.py --loss pg` | reverse-KL starts low, converges; adapter saved → `hf_mopd.md` |
| Same with exact reverse-KL form | `hf_mopd.py --loss kl` | comparable dynamics → `hf_mopd.md` |
| One student holds BOTH abilities (single seed) | `hf_evaluate.py` | MOPD avg normalised score ≈ 1 on both domains → `hf_evaluate.md` |
| **Robust across seeds (headline)** | `hf_seeds.py --seeds 0 1 2` | MOPD ≥1 on both domains every seed; mopd_pg 1.07±0.12, mopd_kl 1.06±0.09 avg → `hf_seeds.md` |

Infra smoke (loads the model; needs GPU): `python experiments/mopd/mopd_hf.py`.

## Motivation / negative control — from-scratch RL is too weak

Why a pretrained base: on FactWorld's tasks, outcome-RL barely specialises a **from-scratch**
tiny transformer (these tasks are supervision-density-limited — the repo's own thesis; cf. the
archived s5-GRPO negative result). That is what motivated the pivot. This path is preserved:

| Claim | Script | Expected headline |
|---|---|---|
| Infra correctness (losses, routing, ckpt round-trip) — CPU-ok | `python tests/test_mopd.py` | all smoke checks PASS |
| A weak shared base with RL headroom on both domains | `stage1_base.py --steps 15000` | binding/recall ≈0.4, pass@8≫greedy → `stage1_base.md`, `probe_headroom.md` |
| Chain is an RL **wall**; binding/recall lift only marginally from scratch | `stage2_teachers.py --lr 3e-4` | teacher−base small/non-monotone → `stage2_teachers.md` |

## Notes

- **Same-origin invariant.** All Qwen adapters share one frozen backbone → initial
  student↔teacher KL ≈ 0 (LoRA zero-init), the stability condition the paper shows is
  load-bearing. `mopd_hf.py` self-check prints this KL (≈0).
- **No thinking.** Qwen rollouts run with `enable_thinking=False`: answers are a clean 1–2-token
  span (fast RL, verifiable reward). Thinking raises `contains` accuracy but breaks the clean
  span and makes rollouts ~20× longer (see `bench_qwen.py --think 1`).
- **Domains** and difficulty live in `mopd_hf.DOMAINS` (binding = last-write-wins state; recall =
  associative retrieval under a large distractor pool) — both partial on the base.

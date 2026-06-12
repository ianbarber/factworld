# FactWorld

An **oracle-validated instrument** for studying **capability composition** in sequence models. FactWorld
crosses **recall × state-tracking × {parametric, in-context} knowledge** on a single deterministic world
and — unlike single-capability probes (MQAR, the S5 word problem) — **composes** them in one query (resolve
a stateful binding *and then* recall a property). Labels are correct by construction: a symbolic **oracle**
produces every gold answer and the label path cannot import the renderer, so nothing can leak from text. A
**validity gate** certifies that no shallow baseline (majority / recency / first-position) clears floor on
any task.

📄 **The paper:** [`paper.pdf`](paper.pdf) (typeset PDF) · [`paper.md`](paper.md) (Markdown source) — *FactWorld: An Oracle-Validated Instrument for Composing
Recall, State-Tracking, and Knowledge.* It characterizes a **composition gap** — binding × in-context
recall floors at small scale unless the recall leg degenerates to memorization — and shows the gap is a
*gated joint circuit*: in-context recall is strictly conditioned on resolving the binding, so it is learned
all-or-nothing rather than as a product of two legs. An architecture lever moves it: at ~45M params a
product-structured GatedDeltaProduct hybrid crosses the composite where a param-matched transformer floors.
Reference numbers live in [`docs/results.md`](docs/results.md) and
[`docs/results-ci.md`](docs/results-ci.md) (4-arch baselines + 3-seed CIs + the recall ablation),
[`docs/state-tracking-results.md`](docs/state-tracking-results.md) (dense S5/A5),
[`docs/scale-results.md`](docs/scale-results.md), and
[`docs/composite-results.md`](docs/composite-results.md) (the composition gap + the decomposition).

## Install

The data / oracle / eval layer is pure-stdlib (no GPU needed). Training the models needs PyTorch + fla:

```bash
uv venv && uv pip install "torch==2.12.*" --torch-backend=auto
uv pip install "flash-linear-attention==0.5.0" numpy pytest
```

Verified on an RTX 3090 (sm_86): torch 2.12.0+cu130, triton 3.7.0, fla 0.5.0.

## Quickstart

```python
from factworld.tasks import CANONICAL, generate, score_exact

spec  = CANONICAL["composite_copy_v1"]      # binding × in-context-copy recall, in one query
train = generate(spec, "train", n=8000)     # deterministic; gold from the oracle (never parsed from text)
test  = generate(spec, "test", length=64)   # held-out OOD-length split, fixed seed
acc   = sum(score_exact(p, e.answer) for p, e in zip(preds, test)) / len(test)
```

```bash
# train + evaluate any task end-to-end (position-strict exact match at each OOD length):
python scripts/run_benchmark.py composite_copy_v1 --arch gdp_hybrid --d_model 320 --steps 8000

python -m factworld.tasks       # suite self-test (determinism + oracle round-trip)
python scripts/validate_suite.py  # validity gate: no shallow shortcut clears floor on any task
```

## The task suite

A frozen, versioned registry (`factworld.tasks.CANONICAL`) with one canonical metric — **position-strict
exact match** of the answer span. Each task carries a `kind`:

**Scored** (`REPORTED`) — recall, state, their composition, the knowledge-source axis, and composition depth:

| task | what it measures | difficulty axis |
|---|---|---|
| `recall_copy_v1` | genuine 1-of-N in-context-copy recall | distractor pool (binding load) |
| `binding_v1` | last-write-wins state (the delta-rule axis) | give-stream length |
| `composite_copy_v1` | binding × in-context-copy recall — the 2-hop composition probe | binding horizon |
| `conflict_v1` | parametric ↔ in-context override (memorized map vs context) | pool size |
| `chain_v1` | depth-*k* pointer chase | composition depth |

**Controls** — `recall_v1`, `composite_v1` (memorized fixed-map versions; positive controls / binding
isolation, degenerate as recall scores). **Experimental** — `s5_v1` (non-abelian Sₖ), `binding_load_v1`
(large working set), `composite_copy_scale_v1` (the exact k=5 configuration used for the paper's §5 scale
result). Only `benchmark`-kind tasks are scored; the registry never presents confounded tasks as peers.

Scale any task to stress larger models via explicit difficulty knobs:

```python
hard = CANONICAL["composite_copy_v1"].scaled(k=64, eval_lengths=(32, 64, 128))
```

## Reproducing the paper

Every headline number maps to one script. The data/oracle/eval layer is pure-stdlib; the training runs
below need a CUDA GPU (validated on an RTX 3090). Scripts that write a `docs/*.md` rebuild it after every
cell (crash-safe); the rest print their tables to stdout (transcribed into the cited doc).

```bash
# Instrument-level guarantee
python scripts/validate_suite.py          # validity gate: no majority/recency/first-pos shortcut clears floor (prints PASS)

# 1-command benchmark entry point (any task / scaled variant)
python scripts/run_benchmark.py composite_copy_v1 --arch gdp_hybrid --d_model 320 --steps 8000

# Section 3.1 — state-tracking dissociation                 -> docs/state-tracking-results.md
python scripts/dense_s5.py --group s5     # S5 matrix: gdp_pure / n_h=1 null / gdn / transformer / gru, 3 seeds
python scripts/dense_s5.py --group a5     # A5 not-S5-specific control panel

# Section 3.2 — recall                                       -> docs/results-ci.md, docs/recall-readout-results.md
python scripts/ci_dissociation.py         # recall_copy_v1 + binding_v1, 4 archs x 3 seeds (pool-2 dissociation CIs)
python scripts/recall_attention_test.py   # attention-free attribution: gdp_pure / gdn_pure / gdp_hybrid across pools 2..8
python scripts/recall_fair.py             # the 1-hop-vs-deferred differential (onehop/defsep/defpad; n_heads 4 vs 8)

# Section 4 — composition gap                                -> docs/results.md, docs/results-ci.md, docs/composite-results.md
python scripts/collect_baselines.py       # 4-arch from-scratch reference baselines, all scored tasks (seed 0)
python scripts/sk_composite.py            # memorization diagnostic + the n_h in {1,2,4} fixed-param mechanism control
python scripts/iso.py                      # the n_h ∈ {1,2,4} product-structure ablation at fixed params (neg-eig on/off)
python scripts/decompose.py               # the gap decomposed: state leg vs recall leg + routing on holder-wrong examples

# Section 5 — scale + the matched LR sweeps                  -> docs/scale-results.md
python scripts/scale_confirm.py           # 45M multi-seed confirmation: gdp 5 / transformer 5 / gdn 3 seeds (default recipe)
python scripts/transformer_lr_sweep.py    # transformer 45M, 5 LRs x 2 seeds  (negative-arm control: 0/10)
python scripts/gdn_lr_sweep.py            # gdn_hybrid 45M, 5 LRs x 2 seeds    (Layer 2: capable-but-LR-fragile, 1/10)
python scripts/gdp_lr_sweep.py            # gdp_hybrid 45M, 5 LRs x 2 seeds    (Layer 2: broad band, 7/10)
python scripts/gdp_confirm_5e4.py         # gdp 45M @ tuned lr 5e-4, 5 seeds   (pins the L16 5/5, L64 3/5 point estimate)
python scripts/gdn_confirm_3e4.py         # gdn 45M @ lr 3e-4, 5 seeds         (W2: 4/5 converge, 1/5 extrapolate)
python scripts/fair_config.py             # W3: transformer n_heads=8+resid (floor survives 0/10) + recurrent short-conv
```

## Repository layout

```
paper.md                  the paper (Markdown source)
paper.pdf                 the typeset PDF        (rebuild: python scripts/build_pdf.py)
docs/results.md           4-arch reference baselines (position-strict exact match)
docs/results-ci.md        3-seed CIs on the dissociating cells + the attention-free recall ablation
docs/state-tracking-results.md  dense-supervised S5/A5 word problem (the §3.1 state-tracking probe)
docs/scale-results.md     the §5 ~45M scale + matched LR sweeps (recurrence vs attention; product broadens the band)
docs/composite-results.md the §4 small-scale composite (memorization diagnostic + n_h control + the decomposition)
docs/related-work.md      related work with verified citations
factworld/                the instrument (torch-free data/oracle/eval + the model zoo)
  world.py, oracle.py     deterministic KB + symbolic ground-truth solver
  render.py               template renderer + its exact inverse parser (no-leak contract)
  tasks.py                the frozen, scalable task registry + canonical metric
  models.py, train.py     transformer / mamba2 / gdp_hybrid / gdn_hybrid / gru on one skeleton
scripts/                  the runnable suite (run_benchmark, validate_suite, collect_baselines, …)
tests/                    oracle, renderer, tokenizer, model-parity, and validity tests
```

The hybrid configuration (`[recurrent, recurrent, attn, recurrent]`, n_h=4, neg-eig) lives in
`factworld/models.py`.

## Tests

```bash
python tests/test_world_oracle.py     # zero-dependency runner
uv run --with pytest pytest -q        # full suite
```

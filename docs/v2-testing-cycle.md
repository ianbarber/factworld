# v2 natural-language testing cycle

This cycle tested three levers on the FactWorld benchmark:

1. **Formatting effect:** Does rendering prompts in clean natural language (`Renderer(natural=True)`)
   help pretrained chat models compared to the canonical v1 atomic-token format?
2. **Architecture effect:** Does adding an FPRM-style weight-tied looped Transformer with causal 1-D
   convolution change local from-scratch learning?
3. **Training-regime effect:** Do local architectures benefit from short convolution, model scale, or
   longer training on v2 natural data?

## 1. Pretrained-model formatting effect (OpenRouter)

### Models and tasks

Nine models were re-evaluated on the `REPORTED` benchmark tasks plus the experimental `s5_v1` task
using the v2 natural renderer:

- `meta-llama/llama-3.2-3b-instruct`
- `meta-llama/llama-3.1-8b-instruct`
- `qwen/qwen-2.5-7b-instruct`
- `qwen/qwen3-32b`
- `meta-llama/llama-3.3-70b-instruct`
- `deepseek/deepseek-chat`
- `openai/gpt-4o-mini`
- `google/gemini-2.5-flash-lite`
- `anthropic/claude-3-haiku`

Tasks: `recall_copy_v1`, `binding_v1`, `composite_copy_v1`, `conflict_v1`, `chain_v1`, and `s5_v1`.
`n = 20` examples per task/length; greedy decoding; `max_new_tokens = 32`.

### v2 natural results — no task-specific format instructions

| model | binding_v1 | chain_v1 | composite_copy_v1 | conflict_v1 | recall_copy_v1 | s5_v1 |
|---|---|---|---|---|---|---|
| llama-3.2-3b-instruct | 0.400 | 0.000 | 0.000 | 0.800 | 0.700 | 0.000 |
| llama-3.1-8b-instruct | 0.400 | 0.000 | 0.000 | 0.700 | 0.600 | 0.000 |
| qwen-2.5-7b-instruct | 0.250 | 0.250 | 0.000 | 1.000 | 0.950 | 0.000 |
| qwen3-32b | 0.250 | 0.350 | 0.000 | 0.600 | 0.350 | 0.000 |
| llama-3.3-70b-instruct | 0.350 | 0.200 | 0.000 | 1.000 | 1.000 | 0.000 |
| deepseek-chat | 0.300 | 0.150 | 0.000 | 1.000 | 1.000 | 0.000 |
| gpt-4o-mini | 0.300 | 0.200 | 0.000 | 1.000 | 1.000 | 0.000 |
| gemini-2.5-flash-lite | 0.300 | 0.250 | 0.000 | 1.000 | 1.000 | 0.000 |
| claude-3-haiku | 0.500 | 0.050 | 0.000 | 0.800 | 0.500 | 0.000 |

### v2 natural results — with composite/S5 format instructions

A second pass added explicit output-format instructions for `composite_copy_v1` ("answer with the
holder's name followed by the requested value, like 'g3 v9'") and `s5_v1` ("answer with only a role
token followed by a period").

| model | binding_v1 | chain_v1 | composite_copy_v1 | conflict_v1 | recall_copy_v1 | s5_v1 |
|---|---|---|---|---|---|---|
| llama-3.2-3b-instruct | 0.350 | 0.000 | 0.100 | 0.800 | 0.700 | 0.350 |
| llama-3.1-8b-instruct | 0.450 | 0.000 | 0.050 | 0.950 | 0.650 | 0.100 |
| qwen-2.5-7b-instruct | 0.300 | 0.250 | 0.050 | 1.000 | 0.950 | 0.100 |
| qwen3-32b | 0.200 | 0.050 | 0.100 | 0.550 | 0.400 | 0.200 |
| llama-3.3-70b-instruct | 0.500 | 0.100 | **0.800** | 1.000 | 1.000 | 0.150 |
| deepseek-chat | 0.300 | 0.200 | 0.000 | 1.000 | 1.000 | 0.300 |
| gpt-4o-mini | 0.350 | 0.200 | 0.000 | 1.000 | 1.000 | 0.150 |
| gemini-2.5-flash-lite | 0.350 | 0.200 | 0.050 | 1.000 | 1.000 | 0.300 |
| claude-3-haiku | 0.400 | 0.050 | 0.000 | 0.800 | 0.500 | **0.400** |

The format prompts recovered meaningful signal on the two tasks where output structure was ambiguous:
`llama-3.3-70b-instruct` jumps from 0.0 to 0.8 on `composite_copy_v1`, and `s5_v1` moves from chance
floor to 0.15–0.40 across several models.

### Comparison to v1 canonical (from `docs/openrouter/results.md`)

| model | binding_v1 v1 | binding_v1 v2 formatted | composite_copy_v1 v1 | composite_copy_v1 v2 formatted | chain_v1 v1 | chain_v1 v2 formatted |
|---|---|---|---|---|---|---|
| llama-3.3-70b-instruct | **0.700** | 0.500 | **0.200** | **0.800** | **0.167** | 0.100 |
| deepseek-chat | **0.467** | 0.300 | **0.600** | 0.000 | **0.200** | 0.200 |
| gpt-4o-mini | **0.567** | 0.350 | **0.167** | 0.000 | 0.067 | 0.200 |
| qwen-2.5-7b-instruct | **0.467** | 0.300 | **0.067** | 0.050 | **0.133** | **0.250** |
| claude-3-haiku | **0.500** | 0.400 | — | 0.000 | — | 0.050 |

(The v1 table is missing a few models; the v2 table is the full 9-model grid.)

### Formatting-effect conclusion

- Natural-language prompts are **not a universal win**. Single-hop tasks (`recall_copy_v1`,
  `conflict_v1`) remain at ceiling, but binding, composition, chain-depth, and S5 tracking are
  generally harder without the regular token boundaries of canonical v1.
- **Output-format instructions matter.** On `composite_copy_v1` and `s5_v1`, models without format
  hints often emit only a value, only a holder name, or a list; with hints, strong models can
  recover to non-trivial exact-match scores.
- Even with format hints, most models still fail on `composite_copy_v1` and `chain_v1`. These are
  real state-tracking/composition failures, not just formatting artifacts.
- **Recommendation:** Keep canonical v1 as the default API-evaluation format. Natural rendering is
  an opt-in experimental variant and should always be paired with explicit output-format instructions.

### Error-analysis note (harness improvements)

A detailed inspection of example-level predictions found four confounders that inflate the
exact-match gap:

1. **Binding query ambiguity.** The natural query `where is {target}?` was interpreted as "list every
   agent that ever held the object." The renderer now uses `who is the final holder of {target}?` in
   natural mode to remove this ambiguity.

2. **List answers.** Some models still enumerate all holders or values (`g0, g2, g4 .`).
   `APIBackend._call_one` now splits on commas and keeps only the last element for exact-match scoring.

3. **Prose prefixes and missing format hints.** Prefixes like `Let's track...` and `The final holder is...`
   are stripped, and composite/S5 format prompts are now standard in the natural-mode API sweep.

4. **Punctuation mismatch in local training.** Natural prompts use attached punctuation (`g4.`), but
   the training target was left in canonical form (`g4 .`). `tasks.py` now renders natural-mode
   answers with attached punctuation, and `runner.py` normalizes both predictions and gold answers
   before scoring.

After these fixes, residual errors on `composite_copy_v1`, `chain_v1`, and `s5_v1` are genuine
reasoning failures: wrong holders, wrong values, or wrong role tokens with no correct token buried
in the answer.

## 2. Local architecture and training effect

### New code

- `factworld/models.py`: Added `FPRM` — a weight-tied looped Transformer block with a depthwise
  causal 1-D convolution (`CausalConv1d`) and RoPE causal attention.
- `factworld/train.py`, `factworld/backends.py`, `scripts/run_benchmark.py`, `scripts/eval_model.py`:
  Fixed tokenizer construction so that v2 natural corpora tokenize without `<unk>`.
- `factworld/backends.py`: Expanded `APIBackend` normalization for natural-mode prose prefixes,
  colons, and comma-separated lists.
- `factworld/render.py`: Natural `state_easy` query changed from `where is ...?` to
  `who is the final holder of ...?`.
- `scripts/sweep_local_v2.py`: New sweep script for local architecture/training ablations on v2
  natural tasks.
- `tests/test_backends.py`, `tests/test_render.py`: Added tests for the natural-mode query and
  API-backend normalization.

### FPRM architecture

```
embedding
   ↓
[ FPRMBlock ] × n_loops   (default n_loops = n_layers)
   ↓
RMSNorm
   ↓
tied LM head
```

Each `FPRMBlock`:

```
x = x + RoPECausalAttention(CausalConv1d(RMSNorm(x)))
x = x + SwiGLU(RMSNorm(x))
```

The block weights are shared across loops, so parameter count is independent of `n_loops`.

### How to run

Train FPRM on v2 natural `composite_copy_v1`:

```bash
.venv-api/bin/python scripts/sweep_local_v2.py \
    --task composite_copy_v1 \
    --arch fprm,gdp_hybrid,gdp_hybrid_shortconv,transformer \
    --steps 5000 --d_model 256 --n_layers 4 --seeds 0 1 2
```

### Initial local sweep: `composite_copy_v1` natural, d=256, L=4, 4k steps

| arch | L16 | L32 | L64 | mean |
|---|---|---|---|---|
| gdp_hybrid | 0.003 | 0.008 | 0.005 | 0.006 |
| gdp_hybrid_shortconv | 0.005 | 0.000 | 0.000 | 0.002 |
| fprm | 0.008 | 0.005 | 0.005 | 0.006 |
| transformer | 0.000 | 0.000 | 0.000 | 0.000 |

All architectures are at the floor at this scale/steps.

### Stronger-regime sweep: `composite_copy_v1` natural, d=512, 20k steps

| arch | L16 | L32 | L64 | mean |
|---|---|---|---|---|
| gdp_hybrid_shortconv | 0.008 | 0.002 | 0.000 | 0.003 |
| transformer | 0.003 | 0.000 | 0.000 | 0.001 |
| fprm (L=4 loops) | 0.018 | 0.010 | 0.005 | 0.011 |
| fprm (L=8 loops) | 0.017 | 0.007 | 0.003 | 0.009 |

Scaling to d=512 and 20k steps lifts all architectures only slightly above floor. FPRM shows a
small edge at L16, but no configuration solves the task. The residual errors are genuine
composition failures, not merely insufficient training time.

### Learnable-variant sweep: `composite_copy_scale_v1` natural, d=256, L=4, 10k steps

This is the easier composite variant (k=5, recall_pool=5) designed so the recall leg is
independently learnable.

| arch | L16 | L64 | mean |
|---|---|---|---|
| gdp_hybrid | 0.982 | 0.285 | 0.633 |
| gdp_hybrid_shortconv | 0.705 | 0.015 | 0.360 |
| fprm | 0.225 | 0.035 | 0.130 |
| transformer | 0.055 | 0.005 | 0.030 |

`gdp_hybrid` dominates in-distribution (L16), but all architectures struggle on the length
extrapolation (L64). FPRM is comparable to the baselines here, not ahead.

### Learnable-variant sweep: `binding_v1` natural, d=256, L=4, 10k steps

| arch | L16 | L32 | L64 | mean |
|---|---|---|---|---|
| fprm | 0.995 | 0.962 | 0.902 | 0.953 |
| gdp_hybrid | 0.997 | 0.773 | 0.750 | 0.840 |
| gdp_hybrid_shortconv | 0.993 | 0.642 | 0.620 | 0.752 |
| transformer | 0.387 | 0.380 | 0.257 | 0.341 |

`binding_v1` cleanly separates architectures. FPRM is best, followed by GDP hybrid. The short-conv
variant helps at L16 but hurts length extrapolation. The transformer baseline barely learns the task.

## 3. Training-regime effect

The local sweep script supports the following ablations:

- `gdp_hybrid` baseline
- `gdp_hybrid_shortconv` = `gdp_hybrid` with `use_short_conv=True`
- `fprm` (default `n_loops = n_layers`)
- `transformer`

These runs use a torch/CUDA environment with `flash-linear-attention` on the local 5090.

## Overall conclusion

- **Formatting is not the bottleneck for pretrained models, but it can hide signal.** Natural prompts
  hurt or don't help on the hardest tasks unless explicit output-format instructions are given. With
  those instructions, strong models show real (but still limited) composition and S5 ability.
- **Architecture clearly matters on learnable tasks.** On `binding_v1` natural, FPRM (0.953 mean)
  outperforms GDP hybrid (0.840), GDP hybrid+shortconv (0.752), and transformer (0.341). The
  short-conv variant gives a near-perfect L16 score but degrades length extrapolation.
- **`composite_copy_v1` natural is beyond the current local capacity.** Even at d=512/20k, all
  architectures remain near floor, and fixing the punctuation mismatch does not rescue it. On the
  easier `composite_copy_scale_v1`, GDP hybrid wins in-distribution (0.982) but everyone fails at
  L64, suggesting composition-length extrapolation is the bottleneck, not model family.
- **Formatting vs. architecture are separable effects.** Pretrained chat models need explicit output
  hints to score on natural composite/S5; from-scratch local models need the right architecture,
  matching input/output punctuation, and an appropriately-scaled task.

## Files

- `results/openrouter_natural_v2_20260624_112300.{json,md}` — OpenRouter v2 natural sweep, no format hints.
- `results/openrouter_natural_v2_formatted_20260624_122435.{json,md}` — OpenRouter v2 natural sweep with composite/S5 format prompts.
- `docs/openrouter/local-v2-sweep.json` — most recent local sweep JSON (overwritten by `sweep_local_v2.py`).
- `results/composite_copy_v1_natural_*.json` — archived `composite_copy_v1` results (d256/4k and d512/20k).
- `results/composite_copy_scale_v1_natural_d256_L4_10k_fixedans.json` — easier composite sweep with natural answers.
- `results/binding_v1_natural_d256_L4_10k_fixedans.json` — binding architecture sweep with natural answers.
- `scripts/sweep_local_v2.py` — local architecture/training sweep entry point.
- `docs/fprm.md` — FPRM architecture note.

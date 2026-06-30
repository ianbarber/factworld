# Natural-language formatting pilot

**Goal:** Test whether the canonical v1 format's detached punctuation and imperative event lines
("`s1 : give o0 to g0 .`") were artificially lowering pretrained-model scores by confusing the
tokenizer/format, rather than by the underlying reasoning difficulty.

**Approach:** Implement an opt-in `natural=True` renderer variant and run small OpenRouter pilots
(n=10) on `composite_copy_v1@L16` and `s5_v1@L32`, comparing against the canonical v1 prompts with
the same output-format instructions.

## Renderer variants tested

| Variant | Fact | Give | Swap | Cycle | Query |
| --- | --- | --- | --- | --- | --- |
| v1 canonical | `g0 's a0 is v0 .` | `s0 : give o0 to g0 .` | `s0 : swap g0 g1 .` | `s0 : cycle the roles of g0 g1 g2 .` | `what is a0 of g0 ?` |
| v2 natural (clean) | `g0's a0 is v0.` | `s0 gives o0 to g0.` | `s0 swaps g0 and g1.` | `s0 cycles roles: g0 -> g1 -> g2.` | `what is a0 of g0?` |
| v2 natural (paraphrase-heavy) | mixed | active/passive mixed | `swaps` / `swaps the roles of` mixed | explicit "X's role goes to Y" | mixed |

The clean variant uses **one fixed phrasing per statement type** (no paraphrase randomisation) and
a compact arrow cycle notation. Gold answers stay in canonical atomic-token form; predictions are
normalised back to that form before scoring.

## Results

### `composite_copy_v1@L16` (n=10)

| model | v1 canonical + format | clean natural + format | paraphrase-heavy natural + format |
| --- | --- | --- | --- |
| moonshotai/kimi-k2 | **0.600** | 0.100 | 0.400 |
| deepseek/deepseek-chat | **0.500** | 0.000 | 0.000 |
| openai/gpt-4o-mini | **0.100** | 0.000 | 0.100 |

### `s5_v1@L32` (n=10)

| model | v1 canonical + format | clean natural + format | paraphrase-heavy natural + format |
| --- | --- | --- | --- |
| moonshotai/kimi-k2 | **0.400** | 0.200 | 0.000 |
| deepseek/deepseek-chat | 0.200 | 0.200 | 0.200 |
| openai/gpt-4o-mini | 0.000 | **0.100** | 0.000 |

### Control: v1 canonical *without* format instruction

Withholding the output-format instruction drove every model to ~0% on both tasks, confirming that
the format instruction is the dominant variable, not the surface grammar of the prompt.

## Take-aways

1. **Format instructions matter most.** The canonical v1 format plus an explicit output-format
   instruction strongly outperforms natural prompts on `composite_copy_v1` and is competitive or
   better on `s5_v1`.

2. **Natural language does not unblock pretrained models.** Moving to attached punctuation and
   subject-verb event lines did not improve scores; if anything it made composition harder for the
   models tested. The core difficulty appears to be state-tracking/composition, not tokenisation
   spacing.

3. **Consistency beats paraphrase variety.** The clean fixed-phrasing natural format performed
   better than the paraphrase-heavy version on S5, suggesting that the earlier natural pilot was
   hurt by inconsistent active/passive and attribute phrasings.

4. **Keep natural as an opt-in variant.** The renderer now supports `natural=True` (clean, fixed
   phrasing) for experiments, but the canonical v1 format remains the default for API evaluation
   and the canonical task registry.

## Recommendation

Do **not** migrate the canonical task format to natural language for API evaluation. The evidence
from this pilot does not support the hypothesis that tokenizer/formatting artifacts are the main
bottleneck for pretrained models on these tasks.

For local training from scratch, a natural corpus remains an interesting ablation, but it would
require retraining and its own grid; the API pilot suggests caution.

## Files

- `docs/openrouter/v1-format-pilot.json` — canonical v1 with format instructions.
- `docs/openrouter/clean-natural-pilot-composite.json` — clean natural, composite.
- `docs/openrouter/clean-natural-pilot-s5-format.json` — clean natural, S5, with format instruction.
- `docs/openrouter/natural-pilot-composite.json` / `natural-pilot-s5.json` — paraphrase-heavy natural.
- `docs/openrouter/v1-baseline-pilot.json` — canonical v1 *without* format instruction (control).

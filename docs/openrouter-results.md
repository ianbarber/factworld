# FactWorld OpenRouter Model Grid

Evaluated at 2026-06-24T02:17:51.311341+00:00.
n = 30 examples per task; position-strict exact match.

System prompt:

> You are taking a short test. Answer each question with only the requested value or values, no explanation. Use the same spelling as in the question.

## Exact-match results

| model | binding_v1 | chain_v1 | composite_copy_v1 | conflict_v1 | recall_copy_v1 |
| --- | --- | --- | --- | --- | --- |
| llama-3.2-3b-instruct | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| llama-3.1-8b-instruct | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| qwen-2.5-7b-instruct | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| qwen3-32b | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| llama-3.3-70b-instruct | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| deepseek-chat | 0.000 | 0.000 | 0.000 | 0.000 | 0.067 |
| gpt-4o-mini | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| gemini-2.5-flash-lite | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| claude-3-haiku | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |

## Relaxed results (whitespace / period invariant)

| model | binding_v1 | chain_v1 | composite_copy_v1 | conflict_v1 | recall_copy_v1 |
| --- | --- | --- | --- | --- | --- |
| llama-3.2-3b-instruct | 0.267 | 0.033 | 0.000 | 0.833 | 0.933 |
| llama-3.1-8b-instruct | 0.233 | 0.033 | 0.000 | 0.767 | 0.867 |
| qwen-2.5-7b-instruct | 0.500 | 0.167 | 0.000 | 0.967 | 1.000 |
| qwen3-32b | 0.367 | 0.333 | 0.000 | 0.300 | 0.500 |
| llama-3.3-70b-instruct | 0.767 | 0.167 | 0.000 | 1.000 | 1.000 |
| deepseek-chat | 0.467 | 0.100 | 0.000 | 1.000 | 0.967 |
| gpt-4o-mini | 0.600 | 0.067 | 0.000 | 1.000 | 1.000 |
| gemini-2.5-flash-lite | 0.067 | 0.300 | 0.000 | 0.933 | 1.000 |
| claude-3-haiku | 0.400 | 0.067 | 0.000 | 0.767 | 0.900 |

## Notes

- Relaxed scoring strips spaces and trailing periods. It is provided because external chat models often emit `v56.` instead of the canonical `v56 .`.
- Exact match remains the canonical FactWorld metric.


## Raw data

```json
[
  {
    "model": "meta-llama/llama-3.2-3b-instruct",
    "task": "recall_copy_v1",
    "length": 6,
    "n": 30,
    "accuracy_exact": 0.0,
    "accuracy_relaxed": 0.9333333333333333,
    "correct_exact": 0,
    "correct_relaxed": 28,
    "elapsed": 2.9959359169006348
  },
  {
    "model": "meta-llama/llama-3.2-3b-instruct",
    "task": "binding_v1",
    "length": 16,
    "n": 30,
    "accuracy_exact": 0.0,
    "accuracy_relaxed": 0.26666666666666666,
    "correct_exact": 0,
    "correct_relaxed": 8,
    "elapsed": 2.5075600147247314
  },
  {
    "model": "meta-llama/llama-3.2-3b-instruct",
    "task": "composite_copy_v1",
    "length": 16,
    "n": 30,
    "accuracy_exact": 0.0,
    "accuracy_relaxed": 0.0,
    "correct_exact": 0,
    "correct_relaxed": 0,
    "elapsed": 2.4316256046295166
  },
  {
    "model": "meta-llama/llama-3.2-3b-instruct",
    "task": "conflict_v1",
    "length": 4,
    "n": 30,
    "accuracy_exact": 0.0,
    "accuracy_relaxed": 0.8333333333333334,
    "correct_exact": 0,
    "correct_relaxed": 25,
    "elapsed": 2.30523419380188
  },
  {
    "model": "meta-llama/llama-3.2-3b-instruct",
    "task": "chain_v1",
    "length": 4,
    "n": 30,
    "accuracy_exact": 0.0,
    "accuracy_relaxed": 0.03333333333333333,
    "correct_exact": 0,
    "correct_relaxed": 1,
    "elapsed": 2.3383915424346924
  },
  {
    "model": "meta-llama/llama-3.1-8b-instruct",
    "task": "recall_copy_v1",
    "length": 6,
    "n": 30,
    "accuracy_exact": 0.0,
    "accuracy_relaxed": 0.8666666666666667,
    "correct_exact": 0,
    "correct_relaxed": 26,
    "elapsed": 6.981614351272583
  },
  {
    "model": "meta-llama/llama-3.1-8b-instruct",
    "task": "binding_v1",
    "length": 16,
    "n": 30,
    "accuracy_exact": 0.0,
    "accuracy_relaxed": 0.23333333333333334,
    "correct_exact": 0,
    "correct_relaxed": 7,
    "elapsed": 3.6952404975891113
  },
  {
    "model": "meta-llama/llama-3.1-8b-instruct",
    "task": "composite_copy_v1",
    "length": 16,
    "n": 30,
    "accuracy_exact": 0.0,
    "accuracy_relaxed": 0.0,
    "correct_exact": 0,
    "correct_relaxed": 0,
    "elapsed": 3.8016655445098877
  },
  {
    "model": "meta-llama/llama-3.1-8b-instruct",
    "task": "conflict_v1",
    "length": 4,
    "n": 30,
    "accuracy_exact": 0.0,
    "accuracy_relaxed": 0.7666666666666667,
    "correct_exact": 0,
    "correct_relaxed": 23,
    "elapsed": 3.445725440979004
  },
  {
    "model": "meta-llama/llama-3.1-8b-instruct",
    "task": "chain_v1",
    "length": 4,
    "n": 30,
    "accuracy_exact": 0.0,
    "accuracy_relaxed": 0.03333333333333333,
    "correct_exact": 0,
    "correct_relaxed": 1,
    "elapsed": 2.8476428985595703
  },
  {
    "model": "qwen/qwen-2.5-7b-instruct",
    "task": "recall_copy_v1",
    "length": 6,
    "n": 30,
    "accuracy_exact": 0.0,
    "accuracy_relaxed": 1.0,
    "correct_exact": 0,
    "correct_relaxed": 30,
    "elapsed": 10.734583139419556
  },
  {
    "model": "qwen/qwen-2.5-7b-instruct",
    "task": "binding_v1",
    "length": 16,
    "n": 30,
    "accuracy_exact": 0.0,
    "accuracy_relaxed": 0.5,
    "correct_exact": 0,
    "correct_relaxed": 15,
    "elapsed": 7.13174295425415
  },
  {
    "model": "qwen/qwen-2.5-7b-instruct",
    "task": "composite_copy_v1",
    "length": 16,
    "n": 30,
    "accuracy_exact": 0.0,
    "accuracy_relaxed": 0.0,
    "correct_exact": 0,
    "correct_relaxed": 0,
    "elapsed": 6.1969404220581055
  },
  {
    "model": "qwen/qwen-2.5-7b-instruct",
    "task": "conflict_v1",
    "length": 4,
    "n": 30,
    "accuracy_exact": 0.0,
    "accuracy_relaxed": 0.9666666666666667,
    "correct_exact": 0,
    "correct_relaxed": 29,
    "elapsed": 6.082268953323364
  },
  {
    "model": "qwen/qwen-2.5-7b-instruct",
    "task": "chain_v1",
    "length": 4,
    "n": 30,
    "accuracy_exact": 0.0,
    "accuracy_relaxed": 0.16666666666666666,
    "correct_exact": 0,
    "correct_relaxed": 5,
    "elapsed": 5.274860382080078
  },
  {
    "model": "qwen/qwen3-32b",
    "task": "recall_copy_v1",
    "length": 6,
    "n": 30,
    "accuracy_exact": 0.0,
    "accuracy_relaxed": 0.5,
    "correct_exact": 0,
    "correct_relaxed": 15,
    "elapsed": 32.7788770198822
  },
  {
    "model": "qwen/qwen3-32b",
    "task": "binding_v1",
    "length": 16,
    "n": 30,
    "accuracy_exact": 0.0,
    "accuracy_relaxed": 0.36666666666666664,
    "correct_exact": 0,
    "correct_relaxed": 11,
    "elapsed": 79.07118344306946
  },
  {
    "model": "qwen/qwen3-32b",
    "task": "composite_copy_v1",
    "length": 16,
    "n": 30,
    "accuracy_exact": 0.0,
    "accuracy_relaxed": 0.0,
    "correct_exact": 0,
    "correct_relaxed": 0,
    "elapsed": 118.75853538513184
  },
  {
    "model": "qwen/qwen3-32b",
    "task": "conflict_v1",
    "length": 4,
    "n": 30,
    "accuracy_exact": 0.0,
    "accuracy_relaxed": 0.3,
    "correct_exact": 0,
    "correct_relaxed": 9,
    "elapsed": 23.808818578720093
  },
  {
    "model": "qwen/qwen3-32b",
    "task": "chain_v1",
    "length": 4,
    "n": 30,
    "accuracy_exact": 0.0,
    "accuracy_relaxed": 0.3333333333333333,
    "correct_exact": 0,
    "correct_relaxed": 10,
    "elapsed": 64.02935910224915
  },
  {
    "model": "meta-llama/llama-3.3-70b-instruct",
    "task": "recall_copy_v1",
    "length": 6,
    "n": 30,
    "accuracy_exact": 0.0,
    "accuracy_relaxed": 1.0,
    "correct_exact": 0,
    "correct_relaxed": 30,
    "elapsed": 4.8102006912231445
  },
  {
    "model": "meta-llama/llama-3.3-70b-instruct",
    "task": "binding_v1",
    "length": 16,
    "n": 30,
    "accuracy_exact": 0.0,
    "accuracy_relaxed": 0.7666666666666667,
    "correct_exact": 0,
    "correct_relaxed": 23,
    "elapsed": 3.8852274417877197
  },
  {
    "model": "meta-llama/llama-3.3-70b-instruct",
    "task": "composite_copy_v1",
    "length": 16,
    "n": 30,
    "accuracy_exact": 0.0,
    "accuracy_relaxed": 0.0,
    "correct_exact": 0,
    "correct_relaxed": 0,
    "elapsed": 4.046652317047119
  },
  {
    "model": "meta-llama/llama-3.3-70b-instruct",
    "task": "conflict_v1",
    "length": 4,
    "n": 30,
    "accuracy_exact": 0.0,
    "accuracy_relaxed": 1.0,
    "correct_exact": 0,
    "correct_relaxed": 30,
    "elapsed": 7.034954786300659
  },
  {
    "model": "meta-llama/llama-3.3-70b-instruct",
    "task": "chain_v1",
    "length": 4,
    "n": 30,
    "accuracy_exact": 0.0,
    "accuracy_relaxed": 0.16666666666666666,
    "correct_exact": 0,
    "correct_relaxed": 5,
    "elapsed": 8.863330364227295
  },
  {
    "model": "deepseek/deepseek-chat",
    "task": "recall_copy_v1",
    "length": 6,
    "n": 30,
    "accuracy_exact": 0.06666666666666667,
    "accuracy_relaxed": 0.9666666666666667,
    "correct_exact": 2,
    "correct_relaxed": 29,
    "elapsed": 8.503005981445312
  },
  {
    "model": "deepseek/deepseek-chat",
    "task": "binding_v1",
    "length": 16,
    "n": 30,
    "accuracy_exact": 0.0,
    "accuracy_relaxed": 0.4666666666666667,
    "correct_exact": 0,
    "correct_relaxed": 14,
    "elapsed": 8.97585153579712
  },
  {
    "model": "deepseek/deepseek-chat",
    "task": "composite_copy_v1",
    "length": 16,
    "n": 30,
    "accuracy_exact": 0.0,
    "accuracy_relaxed": 0.0,
    "correct_exact": 0,
    "correct_relaxed": 0,
    "elapsed": 10.07404375076294
  },
  {
    "model": "deepseek/deepseek-chat",
    "task": "conflict_v1",
    "length": 4,
    "n": 30,
    "accuracy_exact": 0.0,
    "accuracy_relaxed": 1.0,
    "correct_exact": 0,
    "correct_relaxed": 30,
    "elapsed": 7.382169485092163
  },
  {
    "model": "deepseek/deepseek-chat",
    "task": "chain_v1",
    "length": 4,
    "n": 30,
    "accuracy_exact": 0.0,
    "accuracy_relaxed": 0.1,
    "correct_exact": 0,
    "correct_relaxed": 3,
    "elapsed": 9.058337211608887
  },
  {
    "model": "openai/gpt-4o-mini",
    "task": "recall_copy_v1",
    "length": 6,
    "n": 30,
    "accuracy_exact": 0.0,
    "accuracy_relaxed": 1.0,
    "correct_exact": 0,
    "correct_relaxed": 30,
    "elapsed": 5.772373914718628
  },
  {
    "model": "openai/gpt-4o-mini",
    "task": "binding_v1",
    "length": 16,
    "n": 30,
    "accuracy_exact": 0.0,
    "accuracy_relaxed": 0.6,
    "correct_exact": 0,
    "correct_relaxed": 18,
    "elapsed": 4.726925849914551
  },
  {
    "model": "openai/gpt-4o-mini",
    "task": "composite_copy_v1",
    "length": 16,
    "n": 30,
    "accuracy_exact": 0.0,
    "accuracy_relaxed": 0.0,
    "correct_exact": 0,
    "correct_relaxed": 0,
    "elapsed": 5.111829042434692
  },
  {
    "model": "openai/gpt-4o-mini",
    "task": "conflict_v1",
    "length": 4,
    "n": 30,
    "accuracy_exact": 0.0,
    "accuracy_relaxed": 1.0,
    "correct_exact": 0,
    "correct_relaxed": 30,
    "elapsed": 4.769294261932373
  },
  {
    "model": "openai/gpt-4o-mini",
    "task": "chain_v1",
    "length": 4,
    "n": 30,
    "accuracy_exact": 0.0,
    "accuracy_relaxed": 0.06666666666666667,
    "correct_exact": 0,
    "correct_relaxed": 2,
    "elapsed": 5.234834671020508
  },
  {
    "model": "google/gemini-2.5-flash-lite",
    "task": "recall_copy_v1",
    "length": 6,
    "n": 30,
    "accuracy_exact": 0.0,
    "accuracy_relaxed": 1.0,
    "correct_exact": 0,
    "correct_relaxed": 30,
    "elapsed": 4.8544762134552
  },
  {
    "model": "google/gemini-2.5-flash-lite",
    "task": "binding_v1",
    "length": 16,
    "n": 30,
    "accuracy_exact": 0.0,
    "accuracy_relaxed": 0.06666666666666667,
    "correct_exact": 0,
    "correct_relaxed": 2,
    "elapsed": 5.1335227489471436
  },
  {
    "model": "google/gemini-2.5-flash-lite",
    "task": "composite_copy_v1",
    "length": 16,
    "n": 30,
    "accuracy_exact": 0.0,
    "accuracy_relaxed": 0.0,
    "correct_exact": 0,
    "correct_relaxed": 0,
    "elapsed": 3.8385870456695557
  },
  {
    "model": "google/gemini-2.5-flash-lite",
    "task": "conflict_v1",
    "length": 4,
    "n": 30,
    "accuracy_exact": 0.0,
    "accuracy_relaxed": 0.9333333333333333,
    "correct_exact": 0,
    "correct_relaxed": 28,
    "elapsed": 3.7631866931915283
  },
  {
    "model": "google/gemini-2.5-flash-lite",
    "task": "chain_v1",
    "length": 4,
    "n": 30,
    "accuracy_exact": 0.0,
    "accuracy_relaxed": 0.3,
    "correct_exact": 0,
    "correct_relaxed": 9,
    "elapsed": 3.5223734378814697
  },
  {
    "model": "anthropic/claude-3-haiku",
    "task": "recall_copy_v1",
    "length": 6,
    "n": 30,
    "accuracy_exact": 0.0,
    "accuracy_relaxed": 0.9,
    "correct_exact": 0,
    "correct_relaxed": 27,
    "elapsed": 5.05724310874939
  },
  {
    "model": "anthropic/claude-3-haiku",
    "task": "binding_v1",
    "length": 16,
    "n": 30,
    "accuracy_exact": 0.0,
    "accuracy_relaxed": 0.4,
    "correct_exact": 0,
    "correct_relaxed": 12,
    "elapsed": 3.6134257316589355
  },
  {
    "model": "anthropic/claude-3-haiku",
    "task": "composite_copy_v1",
    "length": 16,
    "n": 30,
    "accuracy_exact": 0.0,
    "accuracy_relaxed": 0.0,
    "correct_exact": 0,
    "correct_relaxed": 0,
    "elapsed": 4.280690431594849
  },
  {
    "model": "anthropic/claude-3-haiku",
    "task": "conflict_v1",
    "length": 4,
    "n": 30,
    "accuracy_exact": 0.0,
    "accuracy_relaxed": 0.7666666666666667,
    "correct_exact": 0,
    "correct_relaxed": 23,
    "elapsed": 4.036832571029663
  },
  {
    "model": "anthropic/claude-3-haiku",
    "task": "chain_v1",
    "length": 4,
    "n": 30,
    "accuracy_exact": 0.0,
    "accuracy_relaxed": 0.06666666666666667,
    "correct_exact": 0,
    "correct_relaxed": 2,
    "elapsed": 4.665005922317505
  }
]
```

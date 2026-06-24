# FactWorld OpenRouter Model Grid

Evaluated at 2026-06-24T02:31:58.341545+00:00.
n = 30 examples per task; position-strict exact match.

System prompt:

> You are taking a short test. Answer each question with only the requested value or values, no explanation. Use the same spelling as in the question.

## Exact-match results

| model | binding_v1 | chain_v1 | composite_copy_v1 | conflict_v1 | recall_copy_v1 |
| --- | --- | --- | --- | --- | --- |
| llama-3.2-3b-instruct | 0.267 | 0.033 | 0.000 | 0.833 | 0.933 |
| llama-3.1-8b-instruct | 0.267 | 0.000 | 0.000 | 0.900 | 0.767 |
| qwen-2.5-7b-instruct | 0.500 | 0.167 | 0.000 | 0.967 | 1.000 |
| qwen3-32b | 0.367 | 0.267 | 0.000 | 0.267 | 0.400 |
| llama-3.3-70b-instruct | 0.667 | 0.200 | 0.000 | 1.000 | 1.000 |
| deepseek-chat | 0.567 | 0.267 | 0.000 | 1.000 | 1.000 |
| gpt-4o-mini | 0.633 | 0.067 | 0.000 | 1.000 | 1.000 |
| gemini-2.5-flash-lite | 0.067 | 0.333 | 0.000 | 0.933 | 1.000 |
| claude-3-haiku | 0.400 | 0.067 | 0.000 | 0.733 | 0.900 |

## Relaxed results (whitespace / period invariant)

| model | binding_v1 | chain_v1 | composite_copy_v1 | conflict_v1 | recall_copy_v1 |
| --- | --- | --- | --- | --- | --- |
| llama-3.2-3b-instruct | 0.267 | 0.033 | 0.000 | 0.833 | 0.933 |
| llama-3.1-8b-instruct | 0.267 | 0.000 | 0.000 | 0.900 | 0.767 |
| qwen-2.5-7b-instruct | 0.500 | 0.167 | 0.000 | 0.967 | 1.000 |
| qwen3-32b | 0.367 | 0.267 | 0.000 | 0.267 | 0.400 |
| llama-3.3-70b-instruct | 0.667 | 0.200 | 0.000 | 1.000 | 1.000 |
| deepseek-chat | 0.567 | 0.267 | 0.000 | 1.000 | 1.000 |
| gpt-4o-mini | 0.633 | 0.067 | 0.000 | 1.000 | 1.000 |
| gemini-2.5-flash-lite | 0.067 | 0.333 | 0.000 | 0.933 | 1.000 |
| claude-3-haiku | 0.400 | 0.067 | 0.000 | 0.733 | 0.900 |

## Notes

- `APIBackend` normalizes a trailing period that is glued to the preceding
  token (e.g. `v56.` → `v56 .`), so the canonical exact-match metric is
  meaningful for chat-model output. In this run exact and relaxed scores
  coincide.
- Exact match remains the canonical FactWorld metric.


## Raw data

```json
[
  {
    "model": "meta-llama/llama-3.2-3b-instruct",
    "task": "recall_copy_v1",
    "length": 6,
    "n": 30,
    "accuracy_exact": 0.9333333333333333,
    "accuracy_relaxed": 0.9333333333333333,
    "correct_exact": 28,
    "correct_relaxed": 28,
    "elapsed": 2.8893699645996094
  },
  {
    "model": "meta-llama/llama-3.2-3b-instruct",
    "task": "binding_v1",
    "length": 16,
    "n": 30,
    "accuracy_exact": 0.26666666666666666,
    "accuracy_relaxed": 0.26666666666666666,
    "correct_exact": 8,
    "correct_relaxed": 8,
    "elapsed": 2.074878454208374
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
    "elapsed": 2.5278992652893066
  },
  {
    "model": "meta-llama/llama-3.2-3b-instruct",
    "task": "conflict_v1",
    "length": 4,
    "n": 30,
    "accuracy_exact": 0.8333333333333334,
    "accuracy_relaxed": 0.8333333333333334,
    "correct_exact": 25,
    "correct_relaxed": 25,
    "elapsed": 2.3675639629364014
  },
  {
    "model": "meta-llama/llama-3.2-3b-instruct",
    "task": "chain_v1",
    "length": 4,
    "n": 30,
    "accuracy_exact": 0.03333333333333333,
    "accuracy_relaxed": 0.03333333333333333,
    "correct_exact": 1,
    "correct_relaxed": 1,
    "elapsed": 2.7341816425323486
  },
  {
    "model": "meta-llama/llama-3.1-8b-instruct",
    "task": "recall_copy_v1",
    "length": 6,
    "n": 30,
    "accuracy_exact": 0.7666666666666667,
    "accuracy_relaxed": 0.7666666666666667,
    "correct_exact": 23,
    "correct_relaxed": 23,
    "elapsed": 3.1255593299865723
  },
  {
    "model": "meta-llama/llama-3.1-8b-instruct",
    "task": "binding_v1",
    "length": 16,
    "n": 30,
    "accuracy_exact": 0.26666666666666666,
    "accuracy_relaxed": 0.26666666666666666,
    "correct_exact": 8,
    "correct_relaxed": 8,
    "elapsed": 3.2894163131713867
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
    "elapsed": 3.409355878829956
  },
  {
    "model": "meta-llama/llama-3.1-8b-instruct",
    "task": "conflict_v1",
    "length": 4,
    "n": 30,
    "accuracy_exact": 0.9,
    "accuracy_relaxed": 0.9,
    "correct_exact": 27,
    "correct_relaxed": 27,
    "elapsed": 2.853869915008545
  },
  {
    "model": "meta-llama/llama-3.1-8b-instruct",
    "task": "chain_v1",
    "length": 4,
    "n": 30,
    "accuracy_exact": 0.0,
    "accuracy_relaxed": 0.0,
    "correct_exact": 0,
    "correct_relaxed": 0,
    "elapsed": 5.710058212280273
  },
  {
    "model": "qwen/qwen-2.5-7b-instruct",
    "task": "recall_copy_v1",
    "length": 6,
    "n": 30,
    "accuracy_exact": 1.0,
    "accuracy_relaxed": 1.0,
    "correct_exact": 30,
    "correct_relaxed": 30,
    "elapsed": 9.266985177993774
  },
  {
    "model": "qwen/qwen-2.5-7b-instruct",
    "task": "binding_v1",
    "length": 16,
    "n": 30,
    "accuracy_exact": 0.5,
    "accuracy_relaxed": 0.5,
    "correct_exact": 15,
    "correct_relaxed": 15,
    "elapsed": 7.706951856613159
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
    "elapsed": 11.41635799407959
  },
  {
    "model": "qwen/qwen-2.5-7b-instruct",
    "task": "conflict_v1",
    "length": 4,
    "n": 30,
    "accuracy_exact": 0.9666666666666667,
    "accuracy_relaxed": 0.9666666666666667,
    "correct_exact": 29,
    "correct_relaxed": 29,
    "elapsed": 8.47348141670227
  },
  {
    "model": "qwen/qwen-2.5-7b-instruct",
    "task": "chain_v1",
    "length": 4,
    "n": 30,
    "accuracy_exact": 0.16666666666666666,
    "accuracy_relaxed": 0.16666666666666666,
    "correct_exact": 5,
    "correct_relaxed": 5,
    "elapsed": 9.527146816253662
  },
  {
    "model": "qwen/qwen3-32b",
    "task": "recall_copy_v1",
    "length": 6,
    "n": 30,
    "accuracy_exact": 0.4,
    "accuracy_relaxed": 0.4,
    "correct_exact": 12,
    "correct_relaxed": 12,
    "elapsed": 16.228224992752075
  },
  {
    "model": "qwen/qwen3-32b",
    "task": "binding_v1",
    "length": 16,
    "n": 30,
    "accuracy_exact": 0.36666666666666664,
    "accuracy_relaxed": 0.36666666666666664,
    "correct_exact": 11,
    "correct_relaxed": 11,
    "elapsed": 60.00682711601257
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
    "elapsed": 106.3783872127533
  },
  {
    "model": "qwen/qwen3-32b",
    "task": "conflict_v1",
    "length": 4,
    "n": 30,
    "accuracy_exact": 0.26666666666666666,
    "accuracy_relaxed": 0.26666666666666666,
    "correct_exact": 8,
    "correct_relaxed": 8,
    "elapsed": 17.718878984451294
  },
  {
    "model": "qwen/qwen3-32b",
    "task": "chain_v1",
    "length": 4,
    "n": 30,
    "accuracy_exact": 0.26666666666666666,
    "accuracy_relaxed": 0.26666666666666666,
    "correct_exact": 8,
    "correct_relaxed": 8,
    "elapsed": 32.6307487487793
  },
  {
    "model": "meta-llama/llama-3.3-70b-instruct",
    "task": "recall_copy_v1",
    "length": 6,
    "n": 30,
    "accuracy_exact": 1.0,
    "accuracy_relaxed": 1.0,
    "correct_exact": 30,
    "correct_relaxed": 30,
    "elapsed": 9.172191143035889
  },
  {
    "model": "meta-llama/llama-3.3-70b-instruct",
    "task": "binding_v1",
    "length": 16,
    "n": 30,
    "accuracy_exact": 0.6666666666666666,
    "accuracy_relaxed": 0.6666666666666666,
    "correct_exact": 20,
    "correct_relaxed": 20,
    "elapsed": 7.499256134033203
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
    "elapsed": 7.259193420410156
  },
  {
    "model": "meta-llama/llama-3.3-70b-instruct",
    "task": "conflict_v1",
    "length": 4,
    "n": 30,
    "accuracy_exact": 1.0,
    "accuracy_relaxed": 1.0,
    "correct_exact": 30,
    "correct_relaxed": 30,
    "elapsed": 4.277228832244873
  },
  {
    "model": "meta-llama/llama-3.3-70b-instruct",
    "task": "chain_v1",
    "length": 4,
    "n": 30,
    "accuracy_exact": 0.2,
    "accuracy_relaxed": 0.2,
    "correct_exact": 6,
    "correct_relaxed": 6,
    "elapsed": 3.4695937633514404
  },
  {
    "model": "deepseek/deepseek-chat",
    "task": "recall_copy_v1",
    "length": 6,
    "n": 30,
    "accuracy_exact": 1.0,
    "accuracy_relaxed": 1.0,
    "correct_exact": 30,
    "correct_relaxed": 30,
    "elapsed": 10.504597902297974
  },
  {
    "model": "deepseek/deepseek-chat",
    "task": "binding_v1",
    "length": 16,
    "n": 30,
    "accuracy_exact": 0.5666666666666667,
    "accuracy_relaxed": 0.5666666666666667,
    "correct_exact": 17,
    "correct_relaxed": 17,
    "elapsed": 9.422985792160034
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
    "elapsed": 8.326744794845581
  },
  {
    "model": "deepseek/deepseek-chat",
    "task": "conflict_v1",
    "length": 4,
    "n": 30,
    "accuracy_exact": 1.0,
    "accuracy_relaxed": 1.0,
    "correct_exact": 30,
    "correct_relaxed": 30,
    "elapsed": 8.755098819732666
  },
  {
    "model": "deepseek/deepseek-chat",
    "task": "chain_v1",
    "length": 4,
    "n": 30,
    "accuracy_exact": 0.26666666666666666,
    "accuracy_relaxed": 0.26666666666666666,
    "correct_exact": 8,
    "correct_relaxed": 8,
    "elapsed": 8.6727294921875
  },
  {
    "model": "openai/gpt-4o-mini",
    "task": "recall_copy_v1",
    "length": 6,
    "n": 30,
    "accuracy_exact": 1.0,
    "accuracy_relaxed": 1.0,
    "correct_exact": 30,
    "correct_relaxed": 30,
    "elapsed": 4.867485284805298
  },
  {
    "model": "openai/gpt-4o-mini",
    "task": "binding_v1",
    "length": 16,
    "n": 30,
    "accuracy_exact": 0.6333333333333333,
    "accuracy_relaxed": 0.6333333333333333,
    "correct_exact": 19,
    "correct_relaxed": 19,
    "elapsed": 4.96790885925293
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
    "elapsed": 6.17113733291626
  },
  {
    "model": "openai/gpt-4o-mini",
    "task": "conflict_v1",
    "length": 4,
    "n": 30,
    "accuracy_exact": 1.0,
    "accuracy_relaxed": 1.0,
    "correct_exact": 30,
    "correct_relaxed": 30,
    "elapsed": 6.122320652008057
  },
  {
    "model": "openai/gpt-4o-mini",
    "task": "chain_v1",
    "length": 4,
    "n": 30,
    "accuracy_exact": 0.06666666666666667,
    "accuracy_relaxed": 0.06666666666666667,
    "correct_exact": 2,
    "correct_relaxed": 2,
    "elapsed": 5.28120756149292
  },
  {
    "model": "google/gemini-2.5-flash-lite",
    "task": "recall_copy_v1",
    "length": 6,
    "n": 30,
    "accuracy_exact": 1.0,
    "accuracy_relaxed": 1.0,
    "correct_exact": 30,
    "correct_relaxed": 30,
    "elapsed": 4.500941038131714
  },
  {
    "model": "google/gemini-2.5-flash-lite",
    "task": "binding_v1",
    "length": 16,
    "n": 30,
    "accuracy_exact": 0.06666666666666667,
    "accuracy_relaxed": 0.06666666666666667,
    "correct_exact": 2,
    "correct_relaxed": 2,
    "elapsed": 4.2713096141815186
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
    "elapsed": 3.8910083770751953
  },
  {
    "model": "google/gemini-2.5-flash-lite",
    "task": "conflict_v1",
    "length": 4,
    "n": 30,
    "accuracy_exact": 0.9333333333333333,
    "accuracy_relaxed": 0.9333333333333333,
    "correct_exact": 28,
    "correct_relaxed": 28,
    "elapsed": 3.673326015472412
  },
  {
    "model": "google/gemini-2.5-flash-lite",
    "task": "chain_v1",
    "length": 4,
    "n": 30,
    "accuracy_exact": 0.3333333333333333,
    "accuracy_relaxed": 0.3333333333333333,
    "correct_exact": 10,
    "correct_relaxed": 10,
    "elapsed": 3.6871166229248047
  },
  {
    "model": "anthropic/claude-3-haiku",
    "task": "recall_copy_v1",
    "length": 6,
    "n": 30,
    "accuracy_exact": 0.9,
    "accuracy_relaxed": 0.9,
    "correct_exact": 27,
    "correct_relaxed": 27,
    "elapsed": 4.195646047592163
  },
  {
    "model": "anthropic/claude-3-haiku",
    "task": "binding_v1",
    "length": 16,
    "n": 30,
    "accuracy_exact": 0.4,
    "accuracy_relaxed": 0.4,
    "correct_exact": 12,
    "correct_relaxed": 12,
    "elapsed": 4.747124910354614
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
    "elapsed": 5.663089036941528
  },
  {
    "model": "anthropic/claude-3-haiku",
    "task": "conflict_v1",
    "length": 4,
    "n": 30,
    "accuracy_exact": 0.7333333333333333,
    "accuracy_relaxed": 0.7333333333333333,
    "correct_exact": 22,
    "correct_relaxed": 22,
    "elapsed": 5.262643098831177
  },
  {
    "model": "anthropic/claude-3-haiku",
    "task": "chain_v1",
    "length": 4,
    "n": 30,
    "accuracy_exact": 0.06666666666666667,
    "accuracy_relaxed": 0.06666666666666667,
    "correct_exact": 2,
    "correct_relaxed": 2,
    "elapsed": 4.095459938049316
  }
]
```

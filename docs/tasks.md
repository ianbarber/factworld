# FactWorld task examples

This page shows a concrete example from every canonical task: what the prompt looks like, what the
gold answer is, and — for tasks we ran on the OpenRouter grid — one or two real mistakes made by
strong pretrained models. The goal is to make the difficulty ladder tangible and to show the
difference between a formatting error and a reasoning error.

All gold answers come from the symbolic oracle; none are parsed from rendered text.

---

## `recall_copy_v1` — genuine in-context-copy recall

A random agent→value map is presented in the prompt; the model must copy the queried value.

**Example prompt**

```
g7 has a0 v56 . g5 has a0 v18 . the a0 of g1 is v11 . the a0 of g3 is v40 .
g0 has a0 v41 . g4 's a0 is v16 . what is a0 of g7 ? :
```

**Gold answer:** `v56 .`

**Typical model mistakes** (OpenRouter grid, L6):

| model | prediction | what went wrong |
| --- | --- | --- |
| llama-3.2-3b-instruct | *(empty)* | failed to emit any answer |
| llama-3.1-8b-instruct | `v18 .` | copied the wrong agent's value (binding-load error) |
| qwen3-32b | `.` | emitted only punctuation |

Stronger models (Kimi K2, Nemotron 3 Ultra, GPT-4o-mini) score 1.000 on this task.

---

## `conflict_v1` — parametric vs. in-context override

The model is trained to memorize a fixed agent→value map, but at test time the prompt states a
different value for the queried agent. The correct answer is the *in-context* value.

**Example prompt**

```
g1 has a0 v46 . g12 has a0 v59 . the a0 of g4 is v14 . g3 has a0 v47 .
what is a0 of g12 ? :
```

**Gold answer:** `v59 .`

**Typical model mistakes** (OpenRouter grid, L4):

| model | prediction | what went wrong |
| --- | --- | --- |
| llama-3.2-3b-instruct | *(empty)* | no answer emitted |
| llama-3.1-8b-instruct | `59 .` | correct value but dropped the `v` prefix (tokenizer/formatting) |
| qwen2.5-7b-instruct | `a0 .` | ignored the facts and echoed part of the query |

Most strong models override the memorized map correctly at this pool size.

---

## `binding_v1` — last-write-wins state tracking

A stream of `give` events changes which agent holds each object. The model must report the current
holder of the queried object.

**Example prompt**

```
s0 : o3 is given to g0 . s1 : give o0 to g0 . s2 : give o3 to g3 .
... s14 : o0 is given to g4 . s15 : o1 is given to g4 .
what is the holder of o0 ? :
```

**Gold answer:** `g4 .`

**Typical model mistakes** (OpenRouter grid, L16):

| model | prediction | what went wrong |
| --- | --- | --- |
| llama-3.2-3b-instruct | `g0 .` | returned an earlier holder, not the last write |
| llama-3.1-8b-instruct | `g0, g2, g4 .` | listed multiple holders instead of the current one |
| qwen2.5-7b-instruct | `g1 .` | wrong final holder |

Kimi K2 scores 0.900 here; binding is scale-sensitive but within reach.

---

## `composite_copy_v1` — binding × in-context recall

The model must resolve the current holder of an object (binding) and then recall that agent's value
from the in-context fact map.

**Example prompt**

```
g17 's a0 is v26 . g9 's a0 is v116 . ... g5 's a0 is v85 .
s0 : o3 is given to g30 . s1 : give o2 to g27 . ... s15 : give o0 to g30 .
what is a0 of the holder of o0 ? :
```

**Gold answer:** `g30 v73 .`  (holder = `g30`, value = `v73`)

**Typical model mistakes** (OpenRouter grid, L16):

| model | prediction | what went wrong |
| --- | --- | --- |
| llama-3.2-3b-instruct | `g1 v0 .` | wrong holder and wrong value |
| llama-3.1-8b-instruct | `g1 .` | emitted only the value's agent, not the requested `<holder> <value>` pair |
| qwen2.5-7b-instruct | `g1 .` | same one-token output pattern |

With the composite-format instruction, Nemotron 3 Ultra reaches 0.767 and Kimi K2 0.733. Without
the instruction every model scores 0% because it emits only the value.

---

## `chain_v1` — depth-*k* pointer chase

Each agent points to another agent via an `a0` fact. The model must follow the chain `depth` times.

**Example prompt**

```
g5 's a0 is g2 . g4 has a0 g3 . the a0 of g1 is g5 . g2 has a0 g4 .
the a0 of g0 is g1 . g3 's a0 is g0 . what is a0 of a0 of a0 of a0 of g1 ? :
```

**Gold answer:** `g3 .`  (g1 → g5 → g2 → g4 → g3)

**Typical model mistakes** (OpenRouter grid, L4):

| model | prediction | what went wrong |
| --- | --- | --- |
| llama-3.2-3b-instruct | `g5 .` | stopped after one hop |
| llama-3.1-8b-instruct | `g5 .` | same shallow-chase error |
| qwen2.5-7b-instruct | `g4 .` | off-by-one depth error |

Even the best pretrained models peak at 0.300 on this task.

---

## `s5_v1` — S₅ role-permutation state tracking

A sequence of `swap`/`cycle_roles` events permutes which agent holds each of five roles. The query
asks for the role of a single agent at the end.

**Example prompt**

```
s0 : cycle the roles of g2 g3 g0 g1 g4 . s1 : swap g3 g2 . ...
s31 : cycle the roles of g0 g4 g1 g3 . what role does g0 have ? :
```

**Gold answer:** `r2 .`

**Typical model mistakes** (OpenRouter grid, L32):

| model | prediction | what went wrong |
| --- | --- | --- |
| nemotron-3-ultra-550b-a55b | `r3 .` | one role off; cannot track the running permutation |
| kimi-k2 | `r3 .` | same wrong role on the same example |
| kimi-k2.5 | `r1 .` | different wrong role on a different prompt |

Every model is near the 0.20 chance floor. The format instruction gets the right token *shape*,
but none of the pretrained models tracks the running S₅ permutation.

---

## `recall_v1` — memorized-map recall (control)

The agent→value map is fixed and memorizable. This is a positive control, not a genuine recall test.

**Example prompt**

```
the a0 of g0 is v35 . g1 's a0 is v59 . g2 has a0 v15 . the a0 of g3 is v51 .
the a0 of g4 is v38 . what is a0 of g4 ? :
```

**Gold answer:** `v38 .`

This task is not part of the OpenRouter grid; it is included as a control in the local baseline
suite.

---

## `composite_v1` — binding × memorized recall (control)

The fact map is fixed and memorizable, so this isolates the binding leg of the composite.

**Example prompt**

```
the a0 of g0 is v1 . g1 has a0 v7 . g2 has a0 v52 . g3 has a0 v35 . g4 's a0 is v6 .
s0 : give o1 to g1 . ... s15 : o1 is given to g4 .
what is a0 of the holder of o3 ? :
```

**Gold answer:** `g2 v52 .`

This task is a control, not a scored benchmark; use `composite_copy_v1` for the real composition
probe.

---

## `binding_load_v1` — large working set (experimental)

Eight active objects instead of four. The larger working set exposes interference and is currently
experimental.

**Example prompt**

```
s0 : o5 is given to g2 . s1 : give o0 to g1 . ... s15 : give o4 to g3 .
what is the holder of o5 ? :
```

**Gold answer:** `g4 .`

This task is not in the OpenRouter grid; it is flagged experimental while the dense-supervision
regime is developed.

---

## `composite_copy_scale_v1` — scale-experiment configuration (experimental)

The exact small-pool composite configuration used for the paper's §5 scale experiments. The recall
leg is intentionally easy so that a floor is attributable to composition, not recall capacity.

**Example prompt**

```
g3 has a0 v18 . the a0 of g2 is v29 . the a0 of g1 is v105 . the a0 of g0 is v109 .
g4 has a0 v59 .
s0 : o0 is given to g1 . ... s15 : o3 is given to g0 .
what is a0 of the holder of o3 ? :
```

**Gold answer:** `g0 v109 .`

This task is experimental and excluded from REPORTED; the flagship composition task is
`composite_copy_v1`.

---

## Reading the mistakes

A few patterns show up repeatedly:

- **Formatting collapses to one token.** On `composite_copy_v1`, models often emit only the value
  (or only the holder) unless explicitly told to output `<holder> <value> .`.
- **Recency / first-write shortcuts.** On `binding_v1`, smaller models return an early holder
  instead of the last write.
- **Shallow depth.** On `chain_v1`, models rarely follow more than one or two hops.
- **Chance floor on non-abelian state.** On `s5_v1`, every pretrained model is near the 0.20 floor;
  the mistakes look like random role guesses even though the token shape is correct.

These examples are drawn from the OpenRouter grids in
[`docs/openrouter-results.json`](docs/openrouter-results.json) and
[`docs/openrouter-s5-results.json`](docs/openrouter-s5-results.json).

# s5_bind_v3 — the frontier scout

Written 2026-08-02T10:47:27+00:00. Reading rule pre-registered in `scripts/protocol_s5bind_v3_three_cell_20260731.py` (`scout_verdict`), fixed before any frontier number existed. Metric is **match**, the canonical evaluator, on the answer read only — the frontier arm has no harness-generated checkpoint stream and `assert_trace_read` raises on it.

Informed chance is 1/(k-1) = **0.0909** at the k=12 operating point: the initial map is stated, so the queried agent's own starting value is never the gold answer. **The composed cell has no floor in this regime.** A frontier model reasons in visible tokens, which is a scratchpad, and the composed cell's floor argument is a bound on live slots (W <= max(k,m)+1 = 13 against the task's 25) — the same bound the guided protocol voids locally, where the both-maps replay reaches 0.719 against a printed floor of 0.234. Its number is read against informed chance as a guess baseline and against the other models as a spread.

## Cells

| cell | task | L | n | registered budget | budget it was measured at | prompt tokens |
|---|---|---|---|---|---|---|
| composed@128 | `s5_bind_v3` | 128 | 40 | 16384 | 16384, 32768 | 2453–2699 |
| composed@256 | `s5_bind_v3` | 256 | 40 | 32768 | 32768, 65536, 98304 | 4630–5218 |
| state@85 | `s5_bind_v3_state` | 85 | 40 | 16384 | 16384 | 1181–1379 |
| bind@171 | `s5_bind_v3_bind` | 171 | 40 | 16384 | 16384, 32768 | 1700–2000 |

A cell measured at more than one budget was VOIDED by the truncation rule at the registered one and re-run; the budget history is below. The prompt-token range is across models — the composed prompt alone is ~3.0k tokens at L=128 and ~5.6k at L=256, which is why the 8,192 the scout was originally priced at was a validity defect and not a saving.

## Match, per model per cell

| model | composed@128 | composed@256 | state@85 | bind@171 |
|---|---|---|---|---|
| openai/gpt-5.5 | 1.000 [0.91, 1.00] | 0.975 [0.87, 1.00] | 1.000 [0.91, 1.00] | 1.000 [0.91, 1.00] |
| z-ai/glm-5.2 | 0.575 [0.42, 0.71] | 0.450 [0.31, 0.60] | 0.950 [0.83, 0.99] | 0.850 [0.71, 0.93] |
| nvidia/nemotron-3-ultra-550b-a55b | 0.750 [0.60, 0.86] | 0.500 [0.35, 0.65] | 0.875 [0.74, 0.95] | 1.000 [0.91, 1.00] |

95% Wilson intervals at n=40 in brackets. Per-cell values only; nothing here is averaged across cells or models.

## Validity diagnostics — read before the scores

| model | cell | budget | finish=length | empty | finish reasons | api errors | cost aborted |
|---|---|---|---|---|---|---|---|
| openai/gpt-5.5 | composed@128 | 16384 | 0.00 | 0.00 | `{'stop': 40}` | 0 (+0 finish=error) | False |
| openai/gpt-5.5 | composed@256 | 32768 | 0.00 | 0.00 | `{'stop': 40}` | 0 (+0 finish=error) | False |
| openai/gpt-5.5 | state@85 | 16384 | 0.00 | 0.00 | `{'stop': 40}` | 0 (+0 finish=error) | False |
| openai/gpt-5.5 | bind@171 | 16384 | 0.00 | 0.00 | `{'stop': 40}` | 0 (+0 finish=error) | False |
| z-ai/glm-5.2 | composed@128 | 32768 | 0.07 | 0.07 | `{'stop': 37, 'length': 3}` | 0 (+0 finish=error) | False |
| z-ai/glm-5.2 | composed@256 | 65536 | 0.03 | 0.03 | `{'stop': 39, 'length': 1}` | 0 (+0 finish=error) | False |
| z-ai/glm-5.2 | state@85 | 16384 | 0.03 | 0.03 | `{'stop': 39, 'length': 1}` | 0 (+0 finish=error) | False |
| z-ai/glm-5.2 | bind@171 | 16384 | 0.00 | 0.00 | `{'stop': 40}` | 0 (+0 finish=error) | False |
| nvidia/nemotron-3-ultra-550b-a55b | composed@128 | 32768 | 0.10 | 0.10 | `{'stop': 36, 'length': 4}` | 0 (+0 finish=error) | False |
| nvidia/nemotron-3-ultra-550b-a55b | composed@256 | 98304 | 0.05 | 0.07 | `{'stop': 37, 'length': 2, 'error': 1}` | 0 (+1 finish=error) | False |
| nvidia/nemotron-3-ultra-550b-a55b | state@85 | 16384 | 0.10 | 0.10 | `{'stop': 36, 'length': 4}` | 0 (+0 finish=error) | False |
| nvidia/nemotron-3-ultra-550b-a55b | bind@171 | 32768 | 0.00 | 0.00 | `{'stop': 40}` | 0 (+0 finish=error) | False |

The pre-registered rule evaluates truncation FIRST: any cell over 10% finish=length or empty answers is VOID, is re-run at a raised budget, and enters no stop or buy decision. Evaluated in any other order, a truncated cell reads as a floor — which is how the published s5 L64 cliff was manufactured.

Cells with some truncation but AT OR UNDER the bar are admissible and are read as measured, with the caveat that a truncated call is scored wrong, so their match is a lower bound: z-ai/glm-5.2 composed@128, z-ai/glm-5.2 composed@256, z-ai/glm-5.2 state@85, nvidia/nemotron-3-ultra-550b-a55b composed@128, nvidia/nemotron-3-ultra-550b-a55b composed@256, nvidia/nemotron-3-ultra-550b-a55b state@85.

### The budgets the truncation rule voided, and what they bought

| model | cell | budget | match (scored) | upper bound if every failed call were right | finish=length | empty |
|---|---|---|---|---|---|---|
| z-ai/glm-5.2 | composed@128 | 16384 | 0.500 | 0.775 | 0.28 | 0.28 |
| z-ai/glm-5.2 | composed@256 | 32768 | 0.500 | 0.700 | 0.20 | 0.20 |
| nvidia/nemotron-3-ultra-550b-a55b | bind@171 | 16384 | 0.825 | 0.975 | 0.03 | 0.15 |
| nvidia/nemotron-3-ultra-550b-a55b | composed@128 | 16384 | 0.425 | 0.925 | 0.50 | 0.50 |
| nvidia/nemotron-3-ultra-550b-a55b | composed@256 | 32768 | 0.275 | 0.550 | 0.28 | 0.28 |
| nvidia/nemotron-3-ultra-550b-a55b | composed@256 | 65536 | 0.400 | 0.650 | 0.23 | 0.25 |

A truncated or empty reply is scored wrong, so a void cell's match is a LOWER bound and the column beside it is the upper one. These rows are kept because the raised-budget re-run replaces them in the decision, not in the record: what a budget bought is a fact about that budget.

The upper bounds also settle what the re-runs could have changed about the ranking: the highest value any voided composed@256 cell could have taken is 0.700, below the 0.975 the top model read on a cell with no truncation at all, so no budget raise anywhere in this run could have moved which model is top.

## Spend — every call billed, including the attempts the truncation rule voided

| model | prompt tok | completion tok | reasoning tok | actual $ | of which voided attempts | estimate $ (4k ctok) | worst case $ |
|---|---|---|---|---|---|---|---|
| openai/gpt-5.5 | 398,603 | 1,064,273 | 1,050,676 | $33.92 | $0.00 | $21.45 | $100.56 |
| z-ai/glm-5.2 | 685,086 | 3,259,149 | 2,462,903 | $10.41 | $4.45 | $2.34 | $10.25 |
| nvidia/nemotron-3-ultra-550b-a55b | 1,047,306 | 5,961,632 | 5,826,455 | $13.64 | $7.64 | $1.63 | $7.43 |
| **total** | | | | **$57.98** | $12.08 | $25.42 | $118.24 |

The estimate column is what `benchmark.cost_estimate` prices at 4,000 assumed completion tokens per call; it reads `assumed_output_tokens` and never `max_new_tokens`, so it under-prices a reasoning cell whose traces run long and it is not a budget guard. The worst case — every call spending its whole registered budget — is what the per-cell dollar caps bound the run to, and it is the number the spend was approved against.

## Component floors, recomputed from the exact scored items

| cell | floor on the 40 scored items | floor on 4,000 disjoint items | operative | basis | gpt-5.5 | glm-5.2 | nemotron-3-ultra-550b-a55b |
|---|---|---|---|---|---|---|---|
| state@85 | 0.1500 | 0.0910 | **0.1500** | measured | 1.000 | 0.950 | 0.875 |
| bind@171 | 0.0909 | 0.0909 | **0.0909** | chance | 1.000 | 0.850 | 1.000 |

The two differ where the max over admitted rows takes a high draw at n=40; the larger is operative. The component floors survive a scratchpad by structure: their admitted rows are depth <= 1 and cost under the cell's own algorithm's per-item minimum, and a pad substitutes for registers, not for chaining. The composed cell has no row here because it has no floor in this regime.

## The verdict

- 1. VOID (truncation), FIRST: any cell with more than 10% finish=length or empty answers is VOID — re-run that cell at a raised budget, and it may not enter any stop or buy decision. Without this, STOP(floor) fires on truncation and manufactures the published s5 L64 cliff a second time.
- 2. STOP (ceiling) if the top scout model's COMPOSED match >= 0.9 at L=256. A ceiling cannot rank the roster, so the roster run buys nothing; redesign (raise k or L) before spending.
- 3. STOP (floor) if all three models are within 2 se of INFORMED CHANCE 1/(k-1) = 0.0909 on composed@128. Stated against informed chance and NOT against an operative floor: the frontier is a scratchpad regime, so the composed cell is unfloorable there for exactly the reason it is unfloorable on the guided read. Redesign, do not re-budget.
- 4. STOP (component) if either COMPONENT is below 0.8 for the top scout model. The composed cell is then state-limited or retrieval-limited and unreadable for the same reason it is unreadable locally when a component does not form.
- 5. BUY the roster run iff the composed cell's spread across the three scout models is >= 0.2 match at either composed length AND both components are >= 0.8 for every scout model — the composed cell discriminates inside the roster's range while the components do not. It is a SPREAD rule and needs no floor, which is why it survives the composed cell's floor retraction; floor-clearing language is banned for the composed cell anywhere in the scout or roster report (SCOUT_COMPOSED_FLOOR_LANGUAGE, asserted on the report text).

**STOP_CEILING** — the top scout model (openai/gpt-5.5) reads 0.975 on composed@256, at or above 0.9. A ceiling cannot rank the roster, so the roster run buys nothing; redesign (raise k or L) before spending.

Top model by composed@256: `openai/gpt-5.5`. Composed spread per length: {128: 0.425, 256: 0.525} against the 0.2 bar. Informed-chance band at 2 se: [0.0, 0.1818]. Components at or above 0.8 for every scout model: True. Per-component values for the top model: {'state': 1.0, 'bind': 1.0}.

The roster run is not bought by this script under any verdict: the scout reports and stops.

### What the rules below the one that fired would have read

- STOP (floor) reads composed@128 at {'gpt-5.5': 1.0, 'glm-5.2': 0.575, 'nemotron-3-ultra-550b-a55b': 0.75} against the informed-chance band [0.0, 0.1818]. It does not fire: no model is at a guess.
- STOP (component) reads the top model's components at {'state': 1.0, 'bind': 1.0} against 0.8. It does not fire.
- BUY reads a composed spread of {128: 0.425, 256: 0.525} against 0.2, with both components at or above 0.8 for every scout model: True. Its condition is MET.

So the cell is not undiscriminating — it separates the three models by 0.425 at L=128 and 0.525 at L=256, and both components are at or above 0.8 everywhere, which is the buy rule's condition. It is the TOP of the range that is saturated, and a ranking whose first place is a ceiling cannot order the models a roster run is bought to order. That is what the redesign has to move: k or L, not the budget and not the roster.

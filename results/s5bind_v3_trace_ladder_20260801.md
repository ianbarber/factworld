# s5_bind_v3 — the trace ladder, decoded off the saved checkpoints

k=6 · informed chance 1/(k-1) = 0.200 · match · n=128 per cell · GUIDED protocol (events teacher-forced, every per-event checkpoint and the answer generated) · decoded from `results/s5bind_v3_three_cell_depthmatched_20260801_ckpt`, nothing trained.

Two reads of the same gold, both decoded under the GUIDED protocol. **ANSWER** is the emitted answer token — the channel the frontier arm is scored on, though it is scored there under the PLAIN protocol and not this one. **TRACE** is the model's own final checkpoint's value for the queried slot. They score one quantity through two channels (T1, re-measured on the exact scored items in every row below), so a disagreement between them is a disagreement IN A CHANNEL and is reported as that — and, because the protocol is what a floor is priced against, they are read against ONE floor.

A **bold** cell clears its own floor under the pre-registered rule (z > 3.0 and margin >= 0.15, at this read's own n). A † marks a cell with NO FLOOR: the composed cell is unfloorable under this protocol on BOTH channels, because the guided format writes the whole of P then B at every event and so hands out the k + m live slots the one-structure bound prices — to every policy, the task's own algorithm included. It is a property of the protocol, not of the channel: the generated checkpoints accumulate into the same context the answer token is decoded from. **The composed cell's ANSWER floor of 0.234 / 0.227 / 0.211 published in the previous revision of this table is RETRACTED**, along with the bold on the two cells that were read as clearing it. Per-seed values only — this family is bimodal at the emergence threshold, and a mean over one converged and two floored seeds is a number no seed produced.

**Provenance.** The ANSWER read here reproduces `results/s5bind_v3_three_cell_depthmatched_20260801.json` exactly on all 45 cells the two cover, so the decoder is the committed one and the TRACE column is the only new quantity. Where these trace numbers differ from the ad-hoc decode quoted in the previous round's commit message (composed@48 0.836 / 0.953 / 0.867 here against 0.844 / 0.906 / 0.844 there), these are the numbers to use: they come from a decoder that is in the repo, records its batch per row, and re-measures T1 on every scored item — 12672 of 12672 across the 99 cells.

Decode batch is recorded per row. The rows below were scored at guided_batch [32, 128]: the padded batch is a memory knob and not a scoring one (right padding, causal models), and it was measured on one cell at both sizes — bind@62 reads answer 1.000 and trace 1.000 at 32 and at 128, at 127 s against 295 s.

## The roster is not FLOPs-matched

**`gdp_hybrid` IS NOT FLOPs-MATCHED.** It runs 205.67M FLOPs/token against `fprm`'s 165.32M, `transformer`'s 165.28M — a 24% advantage, against this repo's own compute-matching convention (match on FLOPs/token, not on parameters). Every comparison in which `gdp_hybrid` is ahead carries it, and it is the more load-bearing here because `gdp_hybrid` is the only architecture this run reads on the ANSWER channel at all.

## What this decode settles

**Direction.** On the TRACE read the composed cell sits BELOW its work-matched state component on 24 of the 27 (architecture, seed, rung) cells of the registered ladder. The 3 exceptions are +0.055, +0.000, +0.008, and 2 of them are on the architecture whose every cell is at floor, where nothing is interpretable either way.

**The separation does not track depth, so depth is not a usable axis here.** Per seed, the deficit across the three rungs (48/64/96 against 17/23/34) widens monotonically on 1 of the 9 (architecture, seed) pairs (fprm s1) and narrows monotonically on 1 (gdp_hybrid s2); on the rest it is non-monotone. The registered ladder spans 5.7 to 11.3 carrier hops on the composed cell and that range does not move the gap, so a length axis that widens it — if one exists — is not inside the registered grid.

**Neither length nor depth explains the deficit.** The matched-COST control holds on 7 of the 9 seeds: state@80 costs what composed@48 costs on the forward pass and carries 4.7x its depth, and its trace is at or above the composed cell's there. The exceptions are fprm s1 (0.633 against 0.703), transformer s1 (0.102 against 0.172), which are the same runs whose composed cell is not below its state component in the first place.

**The two channels come apart, per run, and that is the round's main finding.** 6 of the 9 (architecture, seed) pairs hold state on their own final checkpoint that they do not emit as an answer, on at least one cell by 0.15 or more: fprm seed 0, fprm seed 1, fprm seed 2, gdp_hybrid seed 2, transformer seed 0, transformer seed 2. On the rest the two reads agree cell for cell. So a floored ANSWER is not evidence about the state, and any null read off the answer channel alone is a statement about emission.

**Headroom exists, and it is on the architecture axis rather than the length axis.** There are 6 (architecture, seed, rung) cells where BOTH the composed cell and its state component are off ceiling and off floor (fprm s0 @48, fprm s0 @64, fprm s0 @96, fprm s1 @48, fprm s1 @64, fprm s1 @96), and none of them is on `gdp_hybrid`, whose state component never leaves ceiling anywhere on this ladder. Lengthening the stream does not open the window; changing the model does.

## The floors, recomputed at n = 128 from each cell's own scored items

One floor per cell, for BOTH channels, because both decode under the same protocol. The plain protocol's number is beside it as a reference — it is what the PLAIN read scores against and it is NOT a bar a guided score cleared.

| cell | guided floor (answer & trace) | basis | pad reach | plain-protocol floor | slot==gold (T1) | copier per-slot | queried slot moves (min/median/max) |
|---|---|---|---|---|---|---|---|
| bind@31 | 0.200 (1.00x) | chance | — | 0.200 (1.00x) | 128/128 | 0.901 | 2/4/9 |
| bind@41 | 0.200 (1.00x) | chance | — | 0.200 (1.00x) | 128/128 | 0.904 | 2/5/12 |
| bind@62 | 0.200 (1.00x) | chance | — | 0.200 (1.00x) | 128/128 | 0.909 | 2/7/14 |
| bind@132 | 0.200 (1.00x) | chance | — | 0.200 (1.00x) | 128/128 | 0.913 | 6/17/27 |
| composed@48 | **unfloorable** | unfloorable | 0.719 | 0.234 (1.17x) | 128/128 | 0.887 | 2/6/14 |
| composed@64 | **unfloorable** | unfloorable | 0.758 | 0.227 (1.13x) | 128/128 | 0.887 | 3/8/17 |
| composed@96 | **unfloorable** | unfloorable | 0.562 | 0.211 (1.05x) | 128/128 | 0.887 | 5/12/23 |
| state@17 | 0.219 (1.09x) | measured | — | 0.200 (1.00x) | 128/128 | 0.804 | 3/6/11 |
| state@23 | 0.206 (1.03x) | measured | — | 0.206 (1.03x) | 128/128 | 0.812 | 3/8/15 |
| state@34 | 0.200 (1.00x) | measured | — | 0.200 (1.00x) | 128/128 | 0.819 | 5/11/18 |
| state@80 | 0.250 (1.25x) | measured | — | 0.250 (1.25x) | 128/128 | 0.827 | 15/26/39 |

The guided floor equals the plain one on every COMPONENT cell and that is the argument, not a coincidence: a component's class rule is depth <= 1 AND cost under that cell's own algorithm's minimum, and a scratchpad buys neither — a pad substitutes for REGISTERS, not for CHAINING. What a scratchpad does buy is the W axis, and no component row needed it. On the COMPOSED cell the W axis is the whole of the registered class's first conjunct and the step conjunct is one the task itself satisfies, so the class that survives contains the task and there is no floor to clear on either channel — a composed-cell score under this protocol is a WITHIN-RUN comparison and never a cleared floor. The pad reach column is what the excluded both-maps class (carry both maps, drop one block of events, replay the rest) scores on the exact items: a lower bound on that class's max, printed so the retracted floor leaves a number rather than a blank.

`ckpt_copy_prev` — emit the previous checkpoint, so the trace never moves — scores **0.000** on the trace at every cell above, because the query gate requires the queried slot to have moved at least twice and to end different from its stated value. The move column is that gate measured rather than asserted: the minimum over the 128 scored items is 2 or more at every cell.

# The registered depth ladder

Every column of a row is the same carrier chain: the composed cell at L against each component at the length carrying the same amount of THAT component's own work (composed@48 holds 17 swaps and 31 gives).

## composed@48 vs state@17 and bind@31 — 5.7 carrier hops

| arch | seed | state@17 answer | state@17 trace | composed@48 answer | composed@48 trace | bind@31 answer | bind@31 trace |
|---|---|---|---|---|---|---|---|
| gdp_hybrid | 0 | **1.000** | **1.000** | 0.836† | 0.836† | **1.000** | **1.000** |
| gdp_hybrid | 1 | **0.984** | **0.992** | 0.953† | 0.953† | **1.000** | **1.000** |
| gdp_hybrid | 2 | 0.219 | **1.000** | 0.289† | 0.867† | **1.000** | **1.000** |
| fprm | 0 | 0.188 | **0.828** | 0.133† | 0.688† | 0.188 | **1.000** |
| fprm | 1 | 0.148 | **0.648** | 0.203† | 0.703† | 0.148 | **1.000** |
| fprm | 2 | 0.180 | 0.336 | 0.125† | 0.203† | 0.148 | **1.000** |
| transformer | 0 | 0.172 | 0.195 | 0.125† | 0.164† | 0.211 | **0.516** |
| transformer | 1 | 0.133 | 0.195 | 0.156† | 0.172† | 0.180 | 0.109 |
| transformer | 2 | 0.211 | 0.172 | 0.133† | 0.164† | 0.141 | **1.000** |
| _floor_ | | 0.219 | 0.219 | unfloorable (pad 0.719) | unfloorable (pad 0.719) | 0.200 | 0.200 |

## composed@64 vs state@23 and bind@41 — 7.7 carrier hops

| arch | seed | state@23 answer | state@23 trace | composed@64 answer | composed@64 trace | bind@41 answer | bind@41 trace |
|---|---|---|---|---|---|---|---|
| gdp_hybrid | 0 | **1.000** | **1.000** | 0.742† | 0.742† | **1.000** | **1.000** |
| gdp_hybrid | 1 | **1.000** | **1.000** | 0.930† | 0.930† | **1.000** | **1.000** |
| gdp_hybrid | 2 | 0.250 | **1.000** | 0.336† | 0.898† | **0.992** | **1.000** |
| fprm | 0 | 0.195 | **0.805** | 0.141† | 0.609† | 0.148 | **1.000** |
| fprm | 1 | 0.133 | **0.664** | 0.195† | 0.625† | 0.164 | **1.000** |
| fprm | 2 | 0.133 | 0.312 | 0.203† | 0.219† | 0.164 | **1.000** |
| transformer | 0 | 0.148 | 0.219 | 0.133† | 0.109† | 0.188 | **0.578** |
| transformer | 1 | 0.164 | 0.164 | 0.188† | 0.164† | 0.141 | 0.148 |
| transformer | 2 | 0.172 | 0.141 | 0.141† | 0.148† | 0.156 | **1.000** |
| _floor_ | | 0.206 | 0.206 | unfloorable (pad 0.758) | unfloorable (pad 0.758) | 0.200 | 0.200 |

## composed@96 vs state@34 and bind@62 — 11.3 carrier hops

| arch | seed | state@34 answer | state@34 trace | composed@96 answer | composed@96 trace | bind@62 answer | bind@62 trace |
|---|---|---|---|---|---|---|---|
| gdp_hybrid | 0 | **1.000** | **1.000** | 0.836† | 0.836† | **1.000** | **1.000** |
| gdp_hybrid | 1 | **0.984** | **0.984** | 0.961† | 0.969† | **0.977** | **1.000** |
| gdp_hybrid | 2 | 0.250 | **0.992** | 0.375† | 0.930† | **1.000** | **1.000** |
| fprm | 0 | 0.125 | **0.812** | 0.188† | 0.703† | 0.125 | **1.000** |
| fprm | 1 | 0.188 | **0.703** | 0.141† | 0.656† | 0.203 | **1.000** |
| fprm | 2 | 0.219 | **0.367** | 0.164† | 0.195† | 0.203 | **1.000** |
| transformer | 0 | 0.094 | 0.242 | 0.188† | 0.148† | 0.172 | **0.453** |
| transformer | 1 | 0.172 | 0.188 | 0.141† | 0.172† | 0.258 | 0.148 |
| transformer | 2 | 0.141 | 0.195 | 0.164† | 0.148† | 0.133 | **0.992** |
| _floor_ | | 0.200 | 0.200 | unfloorable (pad 0.562) | unfloorable (pad 0.562) | 0.200 | 0.200 |

# The matched-COST control

Each component at the length whose FORWARD PASS costs what composed@48 costs — the control that separates "harder because composed" from "harder because longer". Registered at one composed length only: the guided decode is O(n L^2).

## composed@48 vs state@80 and bind@132 — 5.7 carrier hops

| arch | seed | state@80 answer | state@80 trace | composed@48 answer | composed@48 trace | bind@132 answer | bind@132 trace |
|---|---|---|---|---|---|---|---|
| gdp_hybrid | 0 | **0.992** | **0.992** | 0.836† | 0.836† | **0.992** | **1.000** |
| gdp_hybrid | 1 | **0.992** | **0.992** | 0.953† | 0.953† | **0.984** | **1.000** |
| gdp_hybrid | 2 | 0.156 | **1.000** | 0.289† | 0.867† | **1.000** | **1.000** |
| fprm | 0 | 0.156 | **0.766** | 0.133† | 0.688† | 0.141 | **1.000** |
| fprm | 1 | 0.180 | **0.633** | 0.203† | 0.703† | 0.219 | **1.000** |
| fprm | 2 | 0.211 | 0.273 | 0.125† | 0.203† | 0.219 | **0.938** |
| transformer | 0 | 0.164 | 0.203 | 0.125† | 0.164† | 0.203 | **0.414** |
| transformer | 1 | 0.125 | 0.102 | 0.156† | 0.172† | 0.125 | 0.148 |
| transformer | 2 | 0.180 | 0.203 | 0.133† | 0.164† | 0.180 | **0.961** |
| _floor_ | | 0.250 | 0.250 | unfloorable (pad 0.719) | unfloorable (pad 0.719) | 0.200 | 0.200 |

# Does the separation grow with depth?

The composed cell against its WORK-MATCHED state component on the TRACE read, per seed, at each rung of the registered ladder. A negative difference is the composed cell scoring BELOW the state component that carries the same amount of state work. Each row is a 2x2 exact test on that seed's own 128 items against that seed's own 128; the pooled row is labelled pooled and is a secondary statement, because this family is bimodal at the emergence threshold and per-seed values are the result.

| arch | seed | rung | state trace | composed trace | difference | p (this seed) |
|---|---|---|---|---|---|---|
| gdp_hybrid | 0 | composed@48 vs state@17 | 1.000 (128/128) | 0.836 (107/128) | -0.164 | 3.9e-07 |
| gdp_hybrid | 0 | composed@64 vs state@23 | 1.000 (128/128) | 0.742 (95/128) | -0.258 | 2.2e-11 |
| gdp_hybrid | 0 | composed@96 vs state@34 | 1.000 (128/128) | 0.836 (107/128) | -0.164 | 3.9e-07 |
| gdp_hybrid | 1 | composed@48 vs state@17 | 0.992 (127/128) | 0.953 (122/128) | -0.039 | 0.12 |
| gdp_hybrid | 1 | composed@64 vs state@23 | 1.000 (128/128) | 0.930 (119/128) | -0.070 | 0.0034 |
| gdp_hybrid | 1 | composed@96 vs state@34 | 0.984 (126/128) | 0.969 (124/128) | -0.016 | 0.68 |
| gdp_hybrid | 2 | composed@48 vs state@17 | 1.000 (128/128) | 0.867 (111/128) | -0.133 | 8.6e-06 |
| gdp_hybrid | 2 | composed@64 vs state@23 | 1.000 (128/128) | 0.898 (115/128) | -0.102 | 0.00018 |
| gdp_hybrid | 2 | composed@96 vs state@34 | 0.992 (127/128) | 0.930 (119/128) | -0.062 | 0.019 |
| fprm | 0 | composed@48 vs state@17 | 0.828 (106/128) | 0.688 (88/128) | -0.141 | 0.013 |
| fprm | 0 | composed@64 vs state@23 | 0.805 (103/128) | 0.609 (78/128) | -0.195 | 0.00091 |
| fprm | 0 | composed@96 vs state@34 | 0.812 (104/128) | 0.703 (90/128) | -0.109 | 0.057 |
| fprm | 1 | composed@48 vs state@17 | 0.648 (83/128) | 0.703 (90/128) | +0.055 | 0.42 |
| fprm | 1 | composed@64 vs state@23 | 0.664 (85/128) | 0.625 (80/128) | -0.039 | 0.6 |
| fprm | 1 | composed@96 vs state@34 | 0.703 (90/128) | 0.656 (84/128) | -0.047 | 0.5 |
| fprm | 2 | composed@48 vs state@17 | 0.336 (43/128) | 0.203 (26/128) | -0.133 | 0.024 |
| fprm | 2 | composed@64 vs state@23 | 0.312 (40/128) | 0.219 (28/128) | -0.094 | 0.12 |
| fprm | 2 | composed@96 vs state@34 | 0.367 (47/128) | 0.195 (25/128) | -0.172 | 0.0034 |
| transformer | 0 | composed@48 vs state@17 | 0.195 (25/128) | 0.164 (21/128) | -0.031 | 0.63 |
| transformer | 0 | composed@64 vs state@23 | 0.219 (28/128) | 0.109 (14/128) | -0.109 | 0.027 |
| transformer | 0 | composed@96 vs state@34 | 0.242 (31/128) | 0.148 (19/128) | -0.094 | 0.082 |
| transformer | 1 | composed@48 vs state@17 | 0.195 (25/128) | 0.172 (22/128) | -0.023 | 0.75 |
| transformer | 1 | composed@64 vs state@23 | 0.164 (21/128) | 0.164 (21/128) | +0.000 | 1 |
| transformer | 1 | composed@96 vs state@34 | 0.188 (24/128) | 0.172 (22/128) | -0.016 | 0.87 |
| transformer | 2 | composed@48 vs state@17 | 0.172 (22/128) | 0.164 (21/128) | -0.008 | 1 |
| transformer | 2 | composed@64 vs state@23 | 0.141 (18/128) | 0.148 (19/128) | +0.008 | 1 |
| transformer | 2 | composed@96 vs state@34 | 0.195 (25/128) | 0.148 (19/128) | -0.047 | 0.41 |
| gdp_hybrid | _pooled_ | composed@48 vs state@17 | 383/384 | 340/384 | -0.112 | 7.5e-13 |
| gdp_hybrid | _pooled_ | composed@64 vs state@23 | 384/384 | 329/384 | -0.143 | 6.9e-18 |
| gdp_hybrid | _pooled_ | composed@96 vs state@34 | 381/384 | 350/384 | -0.081 | 6.5e-08 |
| fprm | _pooled_ | composed@48 vs state@17 | 232/384 | 204/384 | -0.073 | 0.049 |
| fprm | _pooled_ | composed@64 vs state@23 | 228/384 | 186/384 | -0.109 | 0.003 |
| fprm | _pooled_ | composed@96 vs state@34 | 241/384 | 199/384 | -0.109 | 0.0028 |
| transformer | _pooled_ | composed@48 vs state@17 | 72/384 | 64/384 | -0.021 | 0.51 |
| transformer | _pooled_ | composed@64 vs state@23 | 67/384 | 54/384 | -0.034 | 0.23 |
| transformer | _pooled_ | composed@96 vs state@34 | 80/384 | 60/384 | -0.052 | 0.076 |

# Headroom — where each cell leaves the ceiling

Both cells sat at 0.844-1.000 at the single length the previous round could afford, so headroom was the binding constraint on seeing a larger gap. This is the same read across the registered ladder, per seed: the composed cell at 48 / 64 / 96 beside its work-matched state component at 17 / 23 / 34, which is the same carrier chain at each rung.

| arch | seed | read | state 17 / 23 / 34 | composed 48 / 64 / 96 |
|---|---|---|---|---|
| gdp_hybrid | 0 | answer | 1.000 / 1.000 / 1.000 | 0.836 / 0.742 / 0.836 |
| gdp_hybrid | 0 | trace | 1.000 / 1.000 / 1.000 | 0.836 / 0.742 / 0.836 |
| gdp_hybrid | 1 | answer | 0.984 / 1.000 / 0.984 | 0.953 / 0.930 / 0.961 |
| gdp_hybrid | 1 | trace | 0.992 / 1.000 / 0.984 | 0.953 / 0.930 / 0.969 |
| gdp_hybrid | 2 | answer | 0.219 / 0.250 / 0.250 | 0.289 / 0.336 / 0.375 |
| gdp_hybrid | 2 | trace | 1.000 / 1.000 / 0.992 | 0.867 / 0.898 / 0.930 |
| fprm | 0 | answer | 0.188 / 0.195 / 0.125 | 0.133 / 0.141 / 0.188 |
| fprm | 0 | trace | 0.828 / 0.805 / 0.812 | 0.688 / 0.609 / 0.703 |
| fprm | 1 | answer | 0.148 / 0.133 / 0.188 | 0.203 / 0.195 / 0.141 |
| fprm | 1 | trace | 0.648 / 0.664 / 0.703 | 0.703 / 0.625 / 0.656 |
| fprm | 2 | answer | 0.180 / 0.133 / 0.219 | 0.125 / 0.203 / 0.164 |
| fprm | 2 | trace | 0.336 / 0.312 / 0.367 | 0.203 / 0.219 / 0.195 |
| transformer | 0 | answer | 0.172 / 0.148 / 0.094 | 0.125 / 0.133 / 0.188 |
| transformer | 0 | trace | 0.195 / 0.219 / 0.242 | 0.164 / 0.109 / 0.148 |
| transformer | 1 | answer | 0.133 / 0.164 / 0.172 | 0.156 / 0.188 / 0.141 |
| transformer | 1 | trace | 0.195 / 0.164 / 0.188 | 0.172 / 0.164 / 0.172 |
| transformer | 2 | answer | 0.211 / 0.172 / 0.141 | 0.133 / 0.141 / 0.164 |
| transformer | 2 | trace | 0.172 / 0.141 / 0.195 | 0.164 / 0.148 / 0.148 |

# Is a null a missing rule or a compounding error? (diagnostic, not a score)

The guided read is FREE-RUNNING on the checkpoints: the events are teacher-forced but every slot the model writes is fed back, so one wrong slot is carried into every later checkpoint. A cell at chance under it has two explanations the registered numbers cannot separate — the model never learned the per-event update, or it has the update and its own errors compound away from it.

This is the same slots read under TEACHER FORCING: the gold interleaved document, one forward pass, argmax at each slot position. **moving** is the accuracy on the slots whose value DIFFERS from the previous checkpoint's — the only part a copier does not get for free, and the whole of the per-event update. It is ORACLE-ASSISTED and no verdict reads it: the true history is exactly what the task withholds.

| arch | seed | state@17 | state@80 | bind@31 | composed@48 | composed@96 |
|---|---|---|---|---|---|---|
| gdp_hybrid | 0 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| gdp_hybrid | 1 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| gdp_hybrid | 2 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| fprm | 0 | 1.000 | 1.000 | 1.000 | 0.999 | 0.999 |
| fprm | 1 | 1.000 | 1.000 | 1.000 | 0.999 | 0.999 |
| fprm | 2 | 1.000 | 1.000 | 1.000 | 0.996 | 0.996 |
| transformer | 0 | 0.575 | 0.588 | 0.590 | 0.314 | 0.324 |
| transformer | 1 | 0.575 | 0.580 | 0.000 | 0.243 | 0.240 |
| transformer | 2 | 0.664 | 0.669 | 1.000 | 0.520 | 0.527 |

Read this against the free-running trace in the ladder above, cell for cell. Two things follow and neither is visible on the free-running read alone.

- **`gdp_hybrid` and `fprm` have the per-event update exactly, at every cell** (1.000-1.000 and 0.996-1.000 on the moving slots), including the composed cell at every registered length. Where their free-running trace is nevertheless at floor — every `fprm` seed on the ANSWER read, and `fprm` seed 2 on the trace — the failure is the closed loop, not the rule. So the recipe formed the composition and the read is what loses it.

- **The `transformer`'s null is not an architecture result.** Its moving-slot accuracy is 0.240-0.527 on the composed cell and 0.565-0.669 on the state component, against the other two architectures' 0.996-1.000 at the same width, depth, document set and step count — and its three seeds span 0.000 to 1.000 on the SAME cell (bind@31), one of them holding the retrieval update exactly and one holding none of it. A capacity limit does not give one seed the update exactly and another none of it; that spread is an optimisation outcome. It agrees with the loss: on the checkpoint-document branch of the mixed objective the transformer ends stage 3 at 0.288-0.384 against `gdp_hybrid`'s 0.194-0.203 and `fprm`'s 0.207-0.216, and it is still descending at the last logged step where the other two have plateaued. `fprm` reaches 1.000 at 165.3M FLOPs/token against the transformer's 165.3M, so the budget is not the constraint at this compute. The change the evidence supports is on the optimiser and not the architecture: the seed spread is the largest effect in the arm, and stage 3 restarts warmup-and-cosine from lr 1e-3 for every stage, which is the schedule a d768x8 softmax transformer is the most sensitive member of this roster to. Nothing here supports 'the transformer cannot compose'.

# Where the two channels disagree

Per (arch, seed, cell): the trace read minus the answer read on the SAME items. A positive number is state the model holds and does not emit; a negative one is an answer the model emits without its own final checkpoint carrying it.

| arch | seed | cell | answer | trace | trace - answer |
|---|---|---|---|---|---|
| gdp_hybrid | 0 | bind@31 | 1.000 | 1.000 | +0.000 |
| gdp_hybrid | 0 | bind@41 | 1.000 | 1.000 | +0.000 |
| gdp_hybrid | 0 | bind@62 | 1.000 | 1.000 | +0.000 |
| gdp_hybrid | 0 | bind@132 | 0.992 | 1.000 | +0.008 |
| gdp_hybrid | 0 | composed@48 | 0.836 | 0.836 | +0.000 |
| gdp_hybrid | 0 | composed@64 | 0.742 | 0.742 | +0.000 |
| gdp_hybrid | 0 | composed@96 | 0.836 | 0.836 | +0.000 |
| gdp_hybrid | 0 | state@17 | 1.000 | 1.000 | +0.000 |
| gdp_hybrid | 0 | state@23 | 1.000 | 1.000 | +0.000 |
| gdp_hybrid | 0 | state@34 | 1.000 | 1.000 | +0.000 |
| gdp_hybrid | 0 | state@80 | 0.992 | 0.992 | +0.000 |
| gdp_hybrid | 1 | bind@31 | 1.000 | 1.000 | +0.000 |
| gdp_hybrid | 1 | bind@41 | 1.000 | 1.000 | +0.000 |
| gdp_hybrid | 1 | bind@62 | 0.977 | 1.000 | +0.023 |
| gdp_hybrid | 1 | bind@132 | 0.984 | 1.000 | +0.016 |
| gdp_hybrid | 1 | composed@48 | 0.953 | 0.953 | +0.000 |
| gdp_hybrid | 1 | composed@64 | 0.930 | 0.930 | +0.000 |
| gdp_hybrid | 1 | composed@96 | 0.961 | 0.969 | +0.008 |
| gdp_hybrid | 1 | state@17 | 0.984 | 0.992 | +0.008 |
| gdp_hybrid | 1 | state@23 | 1.000 | 1.000 | +0.000 |
| gdp_hybrid | 1 | state@34 | 0.984 | 0.984 | +0.000 |
| gdp_hybrid | 1 | state@80 | 0.992 | 0.992 | +0.000 |
| gdp_hybrid | 2 | bind@31 | 1.000 | 1.000 | +0.000 |
| gdp_hybrid | 2 | bind@41 | 0.992 | 1.000 | +0.008 |
| gdp_hybrid | 2 | bind@62 | 1.000 | 1.000 | +0.000 |
| gdp_hybrid | 2 | bind@132 | 1.000 | 1.000 | +0.000 |
| gdp_hybrid | 2 | composed@48 | 0.289 | 0.867 | +0.578 |
| gdp_hybrid | 2 | composed@64 | 0.336 | 0.898 | +0.562 |
| gdp_hybrid | 2 | composed@96 | 0.375 | 0.930 | +0.555 |
| gdp_hybrid | 2 | state@17 | 0.219 | 1.000 | +0.781 |
| gdp_hybrid | 2 | state@23 | 0.250 | 1.000 | +0.750 |
| gdp_hybrid | 2 | state@34 | 0.250 | 0.992 | +0.742 |
| gdp_hybrid | 2 | state@80 | 0.156 | 1.000 | +0.844 |
| fprm | 0 | bind@31 | 0.188 | 1.000 | +0.812 |
| fprm | 0 | bind@41 | 0.148 | 1.000 | +0.852 |
| fprm | 0 | bind@62 | 0.125 | 1.000 | +0.875 |
| fprm | 0 | bind@132 | 0.141 | 1.000 | +0.859 |
| fprm | 0 | composed@48 | 0.133 | 0.688 | +0.555 |
| fprm | 0 | composed@64 | 0.141 | 0.609 | +0.469 |
| fprm | 0 | composed@96 | 0.188 | 0.703 | +0.516 |
| fprm | 0 | state@17 | 0.188 | 0.828 | +0.641 |
| fprm | 0 | state@23 | 0.195 | 0.805 | +0.609 |
| fprm | 0 | state@34 | 0.125 | 0.812 | +0.688 |
| fprm | 0 | state@80 | 0.156 | 0.766 | +0.609 |
| fprm | 1 | bind@31 | 0.148 | 1.000 | +0.852 |
| fprm | 1 | bind@41 | 0.164 | 1.000 | +0.836 |
| fprm | 1 | bind@62 | 0.203 | 1.000 | +0.797 |
| fprm | 1 | bind@132 | 0.219 | 1.000 | +0.781 |
| fprm | 1 | composed@48 | 0.203 | 0.703 | +0.500 |
| fprm | 1 | composed@64 | 0.195 | 0.625 | +0.430 |
| fprm | 1 | composed@96 | 0.141 | 0.656 | +0.516 |
| fprm | 1 | state@17 | 0.148 | 0.648 | +0.500 |
| fprm | 1 | state@23 | 0.133 | 0.664 | +0.531 |
| fprm | 1 | state@34 | 0.188 | 0.703 | +0.516 |
| fprm | 1 | state@80 | 0.180 | 0.633 | +0.453 |
| fprm | 2 | bind@31 | 0.148 | 1.000 | +0.852 |
| fprm | 2 | bind@41 | 0.164 | 1.000 | +0.836 |
| fprm | 2 | bind@62 | 0.203 | 1.000 | +0.797 |
| fprm | 2 | bind@132 | 0.219 | 0.938 | +0.719 |
| fprm | 2 | composed@48 | 0.125 | 0.203 | +0.078 |
| fprm | 2 | composed@64 | 0.203 | 0.219 | +0.016 |
| fprm | 2 | composed@96 | 0.164 | 0.195 | +0.031 |
| fprm | 2 | state@17 | 0.180 | 0.336 | +0.156 |
| fprm | 2 | state@23 | 0.133 | 0.312 | +0.180 |
| fprm | 2 | state@34 | 0.219 | 0.367 | +0.148 |
| fprm | 2 | state@80 | 0.211 | 0.273 | +0.062 |
| transformer | 0 | bind@31 | 0.211 | 0.516 | +0.305 |
| transformer | 0 | bind@41 | 0.188 | 0.578 | +0.391 |
| transformer | 0 | bind@62 | 0.172 | 0.453 | +0.281 |
| transformer | 0 | bind@132 | 0.203 | 0.414 | +0.211 |
| transformer | 0 | composed@48 | 0.125 | 0.164 | +0.039 |
| transformer | 0 | composed@64 | 0.133 | 0.109 | -0.023 |
| transformer | 0 | composed@96 | 0.188 | 0.148 | -0.039 |
| transformer | 0 | state@17 | 0.172 | 0.195 | +0.023 |
| transformer | 0 | state@23 | 0.148 | 0.219 | +0.070 |
| transformer | 0 | state@34 | 0.094 | 0.242 | +0.148 |
| transformer | 0 | state@80 | 0.164 | 0.203 | +0.039 |
| transformer | 1 | bind@31 | 0.180 | 0.109 | -0.070 |
| transformer | 1 | bind@41 | 0.141 | 0.148 | +0.008 |
| transformer | 1 | bind@62 | 0.258 | 0.148 | -0.109 |
| transformer | 1 | bind@132 | 0.125 | 0.148 | +0.023 |
| transformer | 1 | composed@48 | 0.156 | 0.172 | +0.016 |
| transformer | 1 | composed@64 | 0.188 | 0.164 | -0.023 |
| transformer | 1 | composed@96 | 0.141 | 0.172 | +0.031 |
| transformer | 1 | state@17 | 0.133 | 0.195 | +0.062 |
| transformer | 1 | state@23 | 0.164 | 0.164 | +0.000 |
| transformer | 1 | state@34 | 0.172 | 0.188 | +0.016 |
| transformer | 1 | state@80 | 0.125 | 0.102 | -0.023 |
| transformer | 2 | bind@31 | 0.141 | 1.000 | +0.859 |
| transformer | 2 | bind@41 | 0.156 | 1.000 | +0.844 |
| transformer | 2 | bind@62 | 0.133 | 0.992 | +0.859 |
| transformer | 2 | bind@132 | 0.180 | 0.961 | +0.781 |
| transformer | 2 | composed@48 | 0.133 | 0.164 | +0.031 |
| transformer | 2 | composed@64 | 0.141 | 0.148 | +0.008 |
| transformer | 2 | composed@96 | 0.164 | 0.148 | -0.016 |
| transformer | 2 | state@17 | 0.211 | 0.172 | -0.039 |
| transformer | 2 | state@23 | 0.172 | 0.141 | -0.031 |
| transformer | 2 | state@34 | 0.141 | 0.195 | +0.055 |
| transformer | 2 | state@80 | 0.180 | 0.203 | +0.023 |

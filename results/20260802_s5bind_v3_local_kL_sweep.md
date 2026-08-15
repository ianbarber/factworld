# s5_bind_v3 on the local arm — the difficulty surface in k and L

Written 2026-08-03T01:41:15+00:00. Model `local/qwen3.6-35b-a3b-nvfp4` (nvidia/Qwen3.6-35B-A3B-NVFP4, 35B total / 3B active, NVFP4, served by vLLM on this machine at a 131,072-token window). Metric is **match**, the canonical evaluator, on the ANSWER only. Effort arm `high`; the served chat template reads an on/off thinking bit and has no effort ladder, so every reasoning rung is this one measurement. n=40 per cell. No paid endpoint was contacted.

**The composed cell has no floor in this regime.** A served reasoning model emits visible tokens, which is a scratchpad, and the composed cell's floor argument bounds LIVE SLOTS (W <= max(k,m)+1 against the task's k+m+1). Its number is read against INFORMED CHANCE 1/(k-1) — the initial map is stated, so the queried agent's own starting value is never gold — which is a guess baseline, not a floor, and which MOVES WITH k. Component cells keep their floors, recomputed below from the exact scored items and from a disjoint pool.

## What the two axes bought

Across the whole k range at one L, match spans {64: 0.15, 128: 0.05, 192: 0.05, 256: 0.125} — against a 95% Wilson half-width of about 0.15 at n=40. Across the whole L range at one k it spans {6: 0.425, 12: 0.475, 24: 0.625, 32: 0.475}.

The completion-token cost runs the OTHER way on the k axis. Tokens per item at the two ends of each axis:

| axis | held fixed | low end | high end |
|---|---|---|---|
| k | L=64 | k=6: 12962 | k=32: 9200 |
| k | L=128 | k=6: 18654 | k=32: 11745 |
| k | L=192 | k=6: 23445 | k=32: 14560 |
| k | L=256 | k=6: 21790 | k=32: 21137 |
| L | k=6 | L=64: 12962 | L=256: 21790 |
| L | k=12 | L=64: 10933 | L=256: 20472 |
| L | k=24 | L=64: 8453 | L=256: 16542 |
| L | k=32 | L=64: 9200 | L=256: 21137 |

So an extra EVENT costs 42–62 completion tokens and 19–20 prompt tokens across the k rungs, and an extra AGENT costs prompt tokens only. That is the price of the axis that moves match; the axis that does not move it is the cheaper one, which is the opposite of the trade the ceiling remedy assumed.

## Validity first — every cell's truncation and empty rate

| cell | k | L | budget | finish=length | empty | finish reasons | api errors | cost aborted | VOID |
|---|---|---|---|---|---|---|---|---|---|
| bind | 6 | 83 | 49152 | 0.00 | 0.00 | `{'stop': 40}` | 0 (+0 finish=error) | False | — |
| bind | 12 | 42 | 32768 | 0.00 | 0.00 | `{'stop': 40}` | 0 (+0 finish=error) | False | — |
| bind | 12 | 85 | 49152 | 0.00 | 0.00 | `{'stop': 40}` | 0 (+0 finish=error) | False | — |
| bind | 12 | 128 | 49152 | 0.00 | 0.00 | `{'stop': 40}` | 0 (+0 finish=error) | False | — |
| bind | 12 | 171 | 65536 | 0.00 | 0.00 | `{'stop': 40}` | 0 (+0 finish=error) | False | — |
| bind | 32 | 86 | 49152 | 0.00 | 0.00 | `{'stop': 40}` | 0 (+0 finish=error) | False | — |
| composed | 6 | 64 | 32768 | 0.05 | 0.05 | `{'stop': 38, 'length': 2}` | 0 (+0 finish=error) | False | — |
| composed | 6 | 128 | 49152 | 0.03 | 0.03 | `{'stop': 39, 'length': 1}` | 0 (+0 finish=error) | False | — |
| composed | 6 | 192 | 65536 | 0.05 | 0.05 | `{'stop': 38, 'length': 2}` | 0 (+0 finish=error) | False | — |
| composed | 6 | 256 | 65536 | 0.00 | 0.00 | `{'stop': 40}` | 0 (+0 finish=error) | False | — |
| composed | 12 | 64 | 32768 | 0.00 | 0.00 | `{'stop': 40}` | 0 (+0 finish=error) | False | — |
| composed | 12 | 128 | 49152 | 0.00 | 0.00 | `{'stop': 40}` | 0 (+0 finish=error) | False | — |
| composed | 12 | 192 | 65536 | 0.00 | 0.00 | `{'stop': 40}` | 0 (+0 finish=error) | False | — |
| composed | 12 | 256 | 65536 | 0.00 | 0.00 | `{'stop': 40}` | 0 (+0 finish=error) | False | — |
| composed | 24 | 64 | 32768 | 0.00 | 0.00 | `{'stop': 40}` | 0 (+0 finish=error) | False | — |
| composed | 24 | 128 | 49152 | 0.03 | 0.03 | `{'stop': 39, 'length': 1}` | 0 (+0 finish=error) | False | — |
| composed | 24 | 192 | 65536 | 0.00 | 0.00 | `{'stop': 40}` | 0 (+0 finish=error) | False | — |
| composed | 24 | 256 | 65536 | 0.00 | 0.00 | `{'stop': 40}` | 0 (+0 finish=error) | False | — |
| composed | 32 | 64 | 32768 | 0.05 | 0.05 | `{'stop': 38, 'length': 2}` | 0 (+0 finish=error) | False | — |
| composed | 32 | 128 | 49152 | 0.03 | 0.03 | `{'stop': 39, 'length': 1}` | 0 (+0 finish=error) | False | — |
| composed | 32 | 192 | 65536 | 0.03 | 0.03 | `{'stop': 39, 'length': 1}` | 0 (+0 finish=error) | False | — |
| composed | 32 | 256 | 65536 | 0.10 | 0.10 | `{'stop': 36, 'length': 4}` | 0 (+0 finish=error) | False | — |
| state | 6 | 45 | 32768 | 0.03 | 0.03 | `{'stop': 39, 'length': 1}` | 0 (+0 finish=error) | False | — |
| state | 12 | 22 | 32768 | 0.05 | 0.05 | `{'stop': 38, 'length': 2}` | 0 (+0 finish=error) | False | — |
| state | 12 | 43 | 32768 | 0.00 | 0.00 | `{'stop': 40}` | 0 (+0 finish=error) | False | — |
| state | 12 | 64 | 32768 | 0.03 | 0.03 | `{'stop': 39, 'length': 1}` | 0 (+0 finish=error) | False | — |
| state | 12 | 85 | 49152 | 0.03 | 0.03 | `{'stop': 39, 'length': 1}` | 0 (+0 finish=error) | False | — |
| state | 32 | 42 | 32768 | 0.03 | 0.03 | `{'stop': 39, 'length': 1}` | 0 (+0 finish=error) | False | — |

A cell over 10% finish=length or empty is VOID and enters no comparison until it is re-run at a raised budget: a truncated call is scored wrong, so a truncated cell reads as a floor. That ordering is the rule, not a preference — the published s5 L64 cliff was a 16-token budget read as a capability.

Cells with some truncation but AT OR UNDER the bar are read as measured, with the caveat that a truncated call is scored wrong, so their match is a LOWER bound: composed@64 k=6 (0.05), composed@128 k=6 (0.03), composed@192 k=6 (0.05), composed@128 k=24 (0.03), composed@64 k=32 (0.05), composed@128 k=32 (0.03), composed@192 k=32 (0.03), composed@256 k=32 (0.10), state@45 k=6 (0.03), state@22 k=12 (0.05), state@64 k=12 (0.03), state@85 k=12 (0.03), state@42 k=32 (0.03).

## The composed surface — match against informed chance 1/(k-1)

| k | chance | L=64 | L=128 | L=192 | L=256 |
|---|---|---|---|---|---|
| 6 | 0.2000 | 0.550 [0.40,0.69] 2.75x z=+5.5 | 0.250 [0.14,0.40] 1.25x z=+0.8 | 0.125 [0.05,0.26] 0.62x z=-1.2 | 0.200 [0.10,0.35] 1.00x z=+0.0 |
| 12 | 0.0909 | 0.550 [0.40,0.69] 6.05x z=+10.1 | 0.250 [0.14,0.40] 2.75x z=+3.5 | 0.075 [0.03,0.20] 0.82x z=-0.4 | 0.075 [0.03,0.20] 0.82x z=-0.4 |
| 24 | 0.0435 | 0.700 [0.55,0.82] 16.10x z=+20.4 | 0.200 [0.10,0.35] 4.60x z=+4.9 | 0.075 [0.03,0.20] 1.73x z=+1.0 | 0.100 [0.04,0.23] 2.30x z=+1.8 |
| 32 | 0.0323 | 0.550 [0.40,0.69] 17.05x z=+18.5 | 0.225 [0.12,0.38] 6.98x z=+6.9 | 0.125 [0.05,0.26] 3.88x z=+3.3 | 0.075 [0.03,0.20] 2.33x z=+1.5 |

95% Wilson intervals in brackets; the multiplier and z are against that row's own informed chance, which is why raw match is not comparable across k rungs. Per-cell values only — nothing is averaged across cells.

## What the difficulty tracks, and what it does not

The cheapest correct algorithm's STATE leg is a chain of `2 n_swap / k` hops (`validity.s5_bind_v3_carrier_hops`): a swap moves two of the k pointers, so the queried agent is touched by that fraction of them. L raises `n_swap` and k DIVIDES it, so the carrier chain is the one quantity BOTH axes move, and it is the obvious candidate for what makes the cell hard. On this model it is not what match tracks.

| k | L | carrier hops | match | x chance | prompt tok/item | completion tok/item |
|---|---|---|---|---|---|---|
| 6 | 64 | 7.33 | 0.550 | 2.75x | 1311 | 12962 |
| 6 | 128 | 15.00 | 0.250 | 1.25x | 2499 | 18654 |
| 6 | 192 | 22.67 | 0.125 | 0.62x | 3713 | 23445 |
| 6 | 256 | 30.33 | 0.200 | 1.00x | 4938 | 21790 |
| 12 | 64 | 3.67 | 0.550 | 6.05x | 1460 | 10933 |
| 12 | 128 | 7.17 | 0.250 | 2.75x | 2656 | 14025 |
| 12 | 192 | 10.67 | 0.075 | 0.82x | 3895 | 17398 |
| 12 | 256 | 14.17 | 0.075 | 0.82x | 5132 | 20472 |
| 24 | 64 | 1.75 | 0.700 | 16.10x | 1803 | 8453 |
| 24 | 128 | 3.58 | 0.200 | 4.60x | 3054 | 13098 |
| 24 | 192 | 5.33 | 0.075 | 1.73x | 4346 | 14821 |
| 24 | 256 | 7.08 | 0.100 | 2.30x | 5629 | 16542 |
| 32 | 64 | 1.38 | 0.550 | 17.05x | 2006 | 9200 |
| 32 | 128 | 2.62 | 0.225 | 6.98x | 3275 | 11745 |
| 32 | 192 | 4.00 | 0.125 | 3.88x | 4578 | 14560 |
| 32 | 256 | 5.38 | 0.075 | 2.33x | 5885 | 21137 |

### The same carrier chain at different L

| carrier hops | cells (k, L) | match at each | spread |
|---|---|---|---|
| 3.58–3.67 | (24, 128), (12, 64) | 0.200, 0.550 | 0.350 |
| 5.33–5.38 | (24, 192), (32, 256) | 0.075, 0.075 | 0.000 |
| 7.08–7.33 | (24, 256), (12, 128), (6, 64) | 0.100, 0.250, 0.550 | 0.450 |
| 14.17–15.00 | (12, 256), (6, 128) | 0.075, 0.250 | 0.175 |

Within a band the carrier chain is the same length to within 10%, and match still spreads by up to 0.450 against a whole-surface span of 0.625; 4 of 4 bands order their cells by L, shorter stream first. So the chain length is not what this model pays for. What it pays for is L — the number of EVENTS it walks — which is the cost of SIMULATING the stream rather than of chasing the carrier through it.

That is the same conclusion the bounded-pad result reached from the other side. A scratchpad substitutes for REGISTERS and not for CHAINING, so k — which prices registers, k+m+1 live slots — is nearly free to a model that writes its working down, while L, which prices the walk, is not.

## The token cost of each axis

| cell | k | L | prompt tok/item | completion tok/item | median | max | budget | wall clock |
|---|---|---|---|---|---|---|---|---|
| bind | 6 | 83 | 926 | 2123 | 1990 | 3911 | 49152 | 60s |
| bind | 12 | 42 | 595 | 1117 | 1167 | 1866 | 32768 | 32s |
| bind | 12 | 85 | 1039 | 1603 | 1469 | 3784 | 49152 | 52s |
| bind | 12 | 128 | 1509 | 1995 | 2032 | 3637 | 49152 | 62s |
| bind | 12 | 171 | 1999 | 2610 | 2780 | 6077 | 65536 | 85s |
| bind | 32 | 86 | 1378 | 1366 | 1281 | 2400 | 49152 | 43s |
| composed | 6 | 64 | 1311 | 12962 | 10895 | 32768 | 32768 | 524s |
| composed | 6 | 128 | 2499 | 18654 | 16127 | 49152 | 49152 | 741s |
| composed | 6 | 192 | 3713 | 23445 | 20287 | 65536 | 65536 | 1117s |
| composed | 6 | 256 | 4938 | 21790 | 20913 | 36083 | 65536 | 839s |
| composed | 12 | 64 | 1460 | 10933 | 9758 | 18408 | 32768 | 363s |
| composed | 12 | 128 | 2656 | 14025 | 13332 | 21193 | 49152 | 443s |
| composed | 12 | 192 | 3895 | 17398 | 16878 | 32326 | 65536 | 625s |
| composed | 12 | 256 | 5132 | 20472 | 20527 | 32523 | 65536 | 757s |
| composed | 24 | 64 | 1803 | 8453 | 8169 | 15314 | 32768 | 263s |
| composed | 24 | 128 | 3054 | 13098 | 11989 | 49152 | 49152 | 530s |
| composed | 24 | 192 | 4346 | 14821 | 15086 | 22950 | 65536 | 501s |
| composed | 24 | 256 | 5629 | 16542 | 17310 | 30515 | 65536 | 626s |
| composed | 32 | 64 | 2006 | 9200 | 7412 | 32768 | 32768 | 435s |
| composed | 32 | 128 | 3275 | 11745 | 10557 | 49152 | 49152 | 530s |
| composed | 32 | 192 | 4578 | 14560 | 14224 | 65536 | 65536 | 696s |
| composed | 32 | 256 | 5885 | 21137 | 17562 | 65536 | 65536 | 1252s |
| state | 6 | 45 | 681 | 11568 | 9182 | 32768 | 32768 | 471s |
| state | 12 | 22 | 454 | 7342 | 5867 | 32768 | 32768 | 310s |
| state | 12 | 43 | 734 | 10982 | 10920 | 23365 | 32768 | 346s |
| state | 12 | 64 | 1013 | 13946 | 12459 | 32768 | 32768 | 505s |
| state | 12 | 85 | 1293 | 17365 | 16149 | 49152 | 49152 | 645s |
| state | 32 | 42 | 1005 | 8658 | 8152 | 32768 | 32768 | 335s |

Reasoning tokens are INSIDE completion tokens on this arm: vLLM's chat-completions usage carries no reasoning-token field and the chat template opens `<think>` in the generation prompt, so the record's `rtok` reads 0 whether or not the model thought. The completion column is therefore the whole token cost of an item, thinking included.

### What one step of each axis costs and buys

| axis | held fixed | step | delta match | delta prompt tok/item | delta completion tok/item |
|---|---|---|---|---|---|
| k | L=64 | 6 -> 12 | +0.000 | +149 | -2029 |
| k | L=64 | 12 -> 24 | +0.150 | +342 | -2480 |
| k | L=64 | 24 -> 32 | -0.150 | +204 | +747 |
| k | L=128 | 6 -> 12 | +0.000 | +157 | -4629 |
| k | L=128 | 12 -> 24 | -0.050 | +398 | -927 |
| k | L=128 | 24 -> 32 | +0.025 | +221 | -1353 |
| k | L=192 | 6 -> 12 | -0.050 | +182 | -6046 |
| k | L=192 | 12 -> 24 | +0.000 | +452 | -2578 |
| k | L=192 | 24 -> 32 | +0.050 | +232 | -260 |
| k | L=256 | 6 -> 12 | -0.125 | +195 | -1319 |
| k | L=256 | 12 -> 24 | +0.025 | +496 | -3929 |
| k | L=256 | 24 -> 32 | -0.025 | +256 | +4595 |
| L | k=6 | 64 -> 128 | -0.300 | +1188 | +5692 |
| L | k=6 | 128 -> 192 | -0.125 | +1214 | +4790 |
| L | k=6 | 192 -> 256 | +0.075 | +1225 | -1655 |
| L | k=12 | 64 -> 128 | -0.300 | +1196 | +3092 |
| L | k=12 | 128 -> 192 | -0.175 | +1239 | +3373 |
| L | k=12 | 192 -> 256 | +0.000 | +1238 | +3073 |
| L | k=24 | 64 -> 128 | -0.500 | +1251 | +4645 |
| L | k=24 | 128 -> 192 | -0.125 | +1292 | +1723 |
| L | k=24 | 192 -> 256 | +0.025 | +1283 | +1721 |
| L | k=32 | 64 -> 128 | -0.325 | +1268 | +2544 |
| L | k=32 | 128 -> 192 | -0.100 | +1303 | +2816 |
| L | k=32 | 192 -> 256 | -0.050 | +1307 | +6577 |

VOID cells are omitted from the deltas rather than differenced against: a truncated cell's match is a lower bound, so a step into or out of one is not a measured step.

## Component cells, against floors recomputed from the exact scored items

| cell | k | L | partner of composed@L | match | floor (scored) | floor (disjoint) | operative | basis | x floor |
|---|---|---|---|---|---|---|---|---|---|
| bind | 6 | 83 | 128 | 1.000 | 0.2000 | 0.2000 | **0.2000** | chance | 5.00x |
| bind | 12 | 42 | 64 | 1.000 | 0.0909 | 0.0909 | **0.0909** | chance | 11.00x |
| bind | 12 | 85 | 128 | 0.975 | 0.0909 | 0.0909 | **0.0909** | chance | 10.72x |
| bind | 12 | 128 | 192 | 0.925 | 0.0909 | 0.0909 | **0.0909** | chance | 10.18x |
| bind | 12 | 171 | 256 | 0.950 | 0.0909 | 0.0909 | **0.0909** | chance | 10.45x |
| bind | 32 | 86 | 128 | 1.000 | 0.0323 | 0.0323 | **0.0323** | chance | 31.00x |
| state | 6 | 45 | 128 | 0.850 | 0.2250 | 0.2000 | **0.2250** | measured | 3.78x |
| state | 12 | 22 | 64 | 0.925 | 0.1250 | 0.0950 | **0.1250** | measured | 7.40x |
| state | 12 | 43 | 128 | 0.875 | 0.1250 | 0.0920 | **0.1250** | measured | 7.00x |
| state | 12 | 64 | 192 | 0.875 | 0.1250 | 0.0909 | **0.1250** | measured | 7.00x |
| state | 12 | 85 | 256 | 0.775 | 0.1500 | 0.0910 | **0.1500** | measured | 5.17x |
| state | 32 | 42 | 128 | 0.825 | 0.1250 | 0.0717 | **0.1250** | measured | 6.60x |

Both floors are printed because they measure different failure modes: the max over admitted rows carries an upward selection bias at n=40, and the house rule is that a floor is recomputed from the items a score is actually read against. The larger is operative. A component floor's admitted rows are depth <= 1 and cost under the cell's own algorithm's per-item minimum, so a scratchpad does not void them — a pad substitutes for registers, not for chaining.

## The composed cell against its own components, at matched work

| k | composed@L | composed match | state@L (match) | bind@L (match) | lower component | composed - lower component |
|---|---|---|---|---|---|---|
| 6 | 128 | 0.250 | 45 (0.850) | 83 (1.000) | 0.850 | -0.600 |
| 12 | 64 | 0.550 | 22 (0.925) | 42 (1.000) | 0.925 | -0.375 |
| 12 | 128 | 0.250 | 43 (0.875) | 85 (0.975) | 0.875 | -0.625 |
| 12 | 192 | 0.075 | 64 (0.875) | 128 (0.925) | 0.875 | -0.800 |
| 12 | 256 | 0.075 | 85 (0.775) | 171 (0.950) | 0.775 | -0.700 |
| 32 | 128 | 0.225 | 42 (0.825) | 86 (1.000) | 0.825 | -0.600 |

The component lengths are the work-matched partners of that composed cell, so each row compares the composed stream against exactly the swaps and exactly the gives it contains. A composed number below both components is not a component deficit; the components are the gate the scout's rule reads before the composed cell means anything.

## Where this model sits against the three scouted frontier models

The scout's numbers are the registered k=12 spec at n=40 on the answer read, which is the same cell and the same read as the k=12 row above.

| model | composed@128 | composed@256 | state@85 | bind@171 |
|---|---|---|---|---|
| openai/gpt-5.5 | 1.000 | 0.975 | 1.000 | 1.000 |
| nvidia/nemotron-3-ultra-550b-a55b | 0.750 | 0.500 | 0.875 | 1.000 |
| z-ai/glm-5.2 | 0.575 | 0.450 | 0.950 | 0.850 |
| **local/qwen3.6-35b-a3b-nvfp4** | 0.250 | 0.075 | 0.775 | 0.950 |

composed@128 0.250 is below the scouted band [0.575, 1.000]; composed@256 0.075 is below the scouted band [0.450, 0.975]; state@85 0.775 is below the scouted band [0.875, 1.000]; bind@171 0.950 is inside the scouted band [0.850, 1.000].

A stand-in has to land inside the band on the cell it is standing in for. On the COMPONENT cells it is close: bind@171 is inside, and state@85 is 0.100 under the bottom of a band whose whole width is 0.125. On the COMPOSED cell it is not: at composed@256 it is at informed chance (0.075 against 0.0909) where the three scouted models span 0.450 to 0.975, and at composed@128 it is 0.325 under the bottom. So this model can stand in for a frontier model on the components and cannot on the composed cell — the axis a composed-cell redesign has to be tested on is exactly the one it is off the bottom of.

The k=6 rung of this sweep is the k=12 frontier spec scaled down and NOT the registered from-scratch cell `s5_bind_local_v3`, which carries match_reads=2 and no q_no_surface gate. Nothing here is a measurement of that cell.

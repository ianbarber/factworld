# s5_bind_v3 — the matched-COST control, completed at all three rungs

k=6 · informed chance 1/(k-1) = 0.200 · match · n=128 per cell · GUIDED protocol (events teacher-forced, every per-event checkpoint and the answer generated) · decoded from `results/s5bind_v3_three_cell_depthmatched_20260801_ckpt`, nothing trained.

The control reads each component at the length whose FORWARD PASS costs what the composed cell's costs, so "harder because composed" is separated from "harder because longer". The previous decode bought it at composed@48 only; the two rungs added here are the ones that give the clause a length axis.

| rung | composed prompt tokens | matched component | its prompt tokens | its work-matched partner |
|---|---|---|---|---|
| composed@48 | 717 | state@80 | 691 | state@17 (187 tokens) |
| composed@64 | 930 | state@108 | 915 | state@23 (235 tokens) |
| composed@96 | 1348 | state@160 | 1331 | state@34 (323 tokens) |

`bind` has no matched-cost partner at 64 or 96: its sampler cannot fill the floor's own item pool past L=144, so those rows are registered ABSENT rather than filled with a shorter cell that does not match the cost.

**Per seed, never a mean** — this family is bimodal at the emergence threshold. A **bold** cell clears its own floor under the pre-registered rule (z > 3.0, margin >= 0.15, at this read's own n); a † marks a cell with NO FLOOR. The composed cell is unfloorable under this protocol on BOTH channels, so every composed column below is † and no row here is a cleared floor: the guided format writes the whole of P then B at every event and hands out the k + m live slots the one-structure bound prices, to every policy including the task's own algorithm.

## The control on the TRACE read

### composed@48 vs state@80 (717 against 691 prompt tokens)

| arch | seed | state@80 | composed@48 | difference | p (this seed) |
|---|---|---|---|---|---|
| gdp_hybrid | 0 | **0.992** (127/128) | 0.836† (107/128) | -0.156 | 4.9e-06 |
| gdp_hybrid | 1 | **0.992** (127/128) | 0.953† (122/128) | -0.039 | 0.12 |
| gdp_hybrid | 2 | **1.000** (128/128) | 0.867† (111/128) | -0.133 | 8.6e-06 |
| fprm | 0 | **0.766** (98/128) | 0.688† (88/128) | -0.078 | 0.21 |
| fprm | 1 | **0.633** (81/128) | 0.703† (90/128) | +0.070 | 0.29 |
| fprm | 2 | 0.273 (35/128) | 0.203† (26/128) | -0.070 | 0.24 |
| transformer | 0 | 0.203 (26/128) | 0.164† (21/128) | -0.039 | 0.52 |
| transformer | 1 | 0.102 (13/128) | 0.172† (22/128) | +0.070 | 0.14 |
| transformer | 2 | 0.203 (26/128) | 0.164† (21/128) | -0.039 | 0.52 |
| _floor_ | | 0.250 | unfloorable (pad 0.719) | | |

### composed@64 vs state@108 (930 against 915 prompt tokens)

| arch | seed | state@108 | composed@64 | difference | p (this seed) |
|---|---|---|---|---|---|
| gdp_hybrid | 0 | **1.000** (128/128) | 0.742† (95/128) | -0.258 | 2.2e-11 |
| gdp_hybrid | 1 | **1.000** (128/128) | 0.930† (119/128) | -0.070 | 0.0034 |
| gdp_hybrid | 2 | **1.000** (128/128) | 0.898† (115/128) | -0.102 | 0.00018 |
| fprm | 0 | **0.805** (103/128) | 0.609† (78/128) | -0.195 | 0.00091 |
| fprm | 1 | **0.602** (77/128) | 0.625† (80/128) | +0.023 | 0.8 |
| fprm | 2 | 0.266 (34/128) | 0.219† (28/128) | -0.047 | 0.47 |
| transformer | 0 | 0.188 (24/128) | 0.109† (14/128) | -0.078 | 0.11 |
| transformer | 1 | 0.188 (24/128) | 0.164† (21/128) | -0.023 | 0.74 |
| transformer | 2 | 0.164 (21/128) | 0.148† (19/128) | -0.016 | 0.86 |
| _floor_ | | 0.242 | unfloorable (pad 0.758) | | |

### composed@96 vs state@160 (1348 against 1331 prompt tokens)

| arch | seed | state@160 | composed@96 | difference | p (this seed) |
|---|---|---|---|---|---|
| gdp_hybrid | 0 | **0.992** (127/128) | 0.836† (107/128) | -0.156 | 4.9e-06 |
| gdp_hybrid | 1 | **0.984** (126/128) | 0.969† (124/128) | -0.016 | 0.68 |
| gdp_hybrid | 2 | **1.000** (128/128) | 0.930† (119/128) | -0.070 | 0.0034 |
| fprm | 0 | **0.406** (52/128) | 0.703† (90/128) | +0.297 | 2.8e-06 |
| fprm | 1 | 0.195 (25/128) | 0.656† (84/128) | +0.461 | 7.3e-14 |
| fprm | 2 | 0.164 (21/128) | 0.195† (25/128) | +0.031 | 0.63 |
| transformer | 0 | 0.164 (21/128) | 0.148† (19/128) | -0.016 | 0.86 |
| transformer | 1 | 0.211 (27/128) | 0.172† (22/128) | -0.039 | 0.53 |
| transformer | 2 | 0.172 (22/128) | 0.148† (19/128) | -0.023 | 0.73 |
| _floor_ | | 0.203 | unfloorable (pad 0.562) | | |

## The control on the ANSWER read

### composed@48 vs state@80 (717 against 691 prompt tokens)

| arch | seed | state@80 | composed@48 | difference | p (this seed) |
|---|---|---|---|---|---|
| gdp_hybrid | 0 | **0.992** (127/128) | 0.836† (107/128) | -0.156 | 4.9e-06 |
| gdp_hybrid | 1 | **0.992** (127/128) | 0.953† (122/128) | -0.039 | 0.12 |
| gdp_hybrid | 2 | 0.156 (20/128) | 0.289† (37/128) | +0.133 | 0.016 |
| fprm | 0 | 0.156 (20/128) | 0.133† (17/128) | -0.023 | 0.72 |
| fprm | 1 | 0.180 (23/128) | 0.203† (26/128) | +0.023 | 0.75 |
| fprm | 2 | 0.211 (27/128) | 0.125† (16/128) | -0.086 | 0.094 |
| transformer | 0 | 0.164 (21/128) | 0.125† (16/128) | -0.039 | 0.48 |
| transformer | 1 | 0.125 (16/128) | 0.156† (20/128) | +0.031 | 0.59 |
| transformer | 2 | 0.180 (23/128) | 0.133† (17/128) | -0.047 | 0.39 |
| _floor_ | | 0.250 | unfloorable (pad 0.719) | | |

### composed@64 vs state@108 (930 against 915 prompt tokens)

| arch | seed | state@108 | composed@64 | difference | p (this seed) |
|---|---|---|---|---|---|
| gdp_hybrid | 0 | **1.000** (128/128) | 0.742† (95/128) | -0.258 | 2.2e-11 |
| gdp_hybrid | 1 | **0.992** (127/128) | 0.930† (119/128) | -0.062 | 0.019 |
| gdp_hybrid | 2 | 0.281 (36/128) | 0.336† (43/128) | +0.055 | 0.42 |
| fprm | 0 | 0.180 (23/128) | 0.141† (18/128) | -0.039 | 0.5 |
| fprm | 1 | 0.117 (15/128) | 0.195† (25/128) | +0.078 | 0.12 |
| fprm | 2 | 0.148 (19/128) | 0.203† (26/128) | +0.055 | 0.32 |
| transformer | 0 | 0.148 (19/128) | 0.133† (17/128) | -0.016 | 0.86 |
| transformer | 1 | 0.141 (18/128) | 0.188† (24/128) | +0.047 | 0.4 |
| transformer | 2 | 0.148 (19/128) | 0.141† (18/128) | -0.008 | 1 |
| _floor_ | | 0.242 | unfloorable (pad 0.758) | | |

### composed@96 vs state@160 (1348 against 1331 prompt tokens)

| arch | seed | state@160 | composed@96 | difference | p (this seed) |
|---|---|---|---|---|---|
| gdp_hybrid | 0 | **0.992** (127/128) | 0.836† (107/128) | -0.156 | 4.9e-06 |
| gdp_hybrid | 1 | **0.984** (126/128) | 0.961† (123/128) | -0.023 | 0.45 |
| gdp_hybrid | 2 | 0.273 (35/128) | 0.375† (48/128) | +0.102 | 0.11 |
| fprm | 0 | 0.148 (19/128) | 0.188† (24/128) | +0.039 | 0.5 |
| fprm | 1 | 0.172 (22/128) | 0.141† (18/128) | -0.031 | 0.61 |
| fprm | 2 | 0.172 (22/128) | 0.164† (21/128) | -0.008 | 1 |
| transformer | 0 | 0.109 (14/128) | 0.188† (24/128) | +0.078 | 0.11 |
| transformer | 1 | 0.148 (19/128) | 0.141† (18/128) | -0.008 | 1 |
| transformer | 2 | 0.141 (18/128) | 0.164† (21/128) | +0.023 | 0.73 |
| _floor_ | | 0.203 | unfloorable (pad 0.562) | | |

## What the completed control settles

**TRACE read.** The matched-cost component is at or above the composed cell on 21 of the 27 (architecture, seed, rung) rows — composed@48 vs state@80: 7/9; composed@64 vs state@108: 8/9; composed@96 vs state@160: 6/9. 5 of those rows have BOTH cells off ceiling and off floor (fprm s0 @48, fprm s0 @64, fprm s0 @96, fprm s1 @48, fprm s1 @64), and a row with both cells pinned is consistent with any direction, so it is counted and not read.

**ANSWER read.** The matched-cost component is at or above the composed cell on 16 of the 27 (architecture, seed, rung) rows — composed@48 vs state@80: 6/9; composed@64 vs state@108: 5/9; composed@96 vs state@160: 5/9. 0 of those rows have BOTH cells off ceiling and off floor (none), and a row with both cells pinned is consistent with any direction, so it is counted and not read.

The exceptions on the trace read are fprm s0 composed@96 0.703 against state@160 0.406, fprm s1 composed@48 0.703 against state@80 0.633, fprm s1 composed@64 0.625 against state@108 0.602, fprm s1 composed@96 0.656 against state@160 0.195, fprm s2 composed@96 0.195 against state@160 0.164, transformer s1 composed@48 0.172 against state@80 0.102.

This is a within-run comparison at every row. The composed cell has no floor under this protocol, so the control can say the deficit is not explained by prompt length; it cannot say the composed cell is above any cheap policy.

## The floors, recomputed at n = 128 from each cell's own scored items

One floor per cell for BOTH channels, because both decode under the same protocol.

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
| state@108 | 0.242 (1.21x) | measured | — | 0.219 (1.09x) | 128/128 | 0.829 | 20/35/49 |
| state@160 | 0.203 (1.02x) | measured | — | 0.200 (1.00x) | 128/128 | 0.830 | 36/54/69 |

Decode batch is recorded per row in the JSON. It is a memory knob and not a scoring one (right padding, causal models).

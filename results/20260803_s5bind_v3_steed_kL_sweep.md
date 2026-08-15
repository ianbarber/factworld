# s5_bind_v3 on steed's DeepSeek V4 — placing the model, and pricing the k axis

Written 2026-08-03T16:18:57+00:00. Model `steed/deepseek-v4-flash` — DeepSeek V4, q2, ~81 GB resident, served by `ds4-server` on a DGX Spark GB10 over the tailnet at a 262,144-token window. Metric is **match**, the canonical evaluator, on the ANSWER only. Effort arm `high`; ds4 collapses `minimal`/`low`/`medium`/`high`/`xhigh` to one internal level, so that is the model's single thinking arm. **No paid endpoint was contacted and the local GPU was not used**; `cost_usd_est` is 0.0 on every record.

**The composed cell has no floor in this regime.** The model emits its working as plain content, which is a scratchpad, and the composed cell's floor argument bounds LIVE SLOTS (W <= max(k,m)+1 against the task's k+m+1). Its number is read against INFORMED CHANCE 1/(k-1) — the initial map is stated, so the queried agent's own starting value is never gold — which is a guess baseline, not a floor, and which MOVES WITH k. Component cells keep their floors, recomputed below from the exact scored items and from a disjoint pool.

## The answers

**k is a WIDTH axis, not a difficulty axis, and the reason is in the task rather than in any model.** Raising k at fixed L divides the pointer chain among more agents: MEASURED off the rendered prompts, the chain at L=128 runs 15.84 touches at k=6 down to 3.19 at k=48. What k does move is the cell's live-slot requirement, 13 slots at k=6 to 97 at k=48. On the local grid, over 16 live cells at n=40, corr(match, log L) = -0.934 while corr(match, log k) = -0.043. A scratchpad buys width with tokens, so k is inert on the FRONTIER arm; on a bounded-state from-scratch architecture width is the binding resource and k is not inert there.

**Worse than inert: past a boundary, raising k stops the cell being a composition cell.** The strongest policy the class rule ADMITS — one structure plus a scratch register, no more steps than the cell's own algorithm — reads 0.99x informed chance at the shipped k=12/L=128 and 10.43x at k=48/L=128. The admissible set is a ray in (k, L) that narrows with k, and on the local grid the margin a measured score holds over that class collapses along k while raw match stays flat.

**So the ceiling's only remaining remedy is L**, with k raised no faster than the ray allows and only for the resolution 1/(k-1) buys. Candidate next paid points and their prices are below, and one probe is priced for decision at the end.

**The free arm cannot substitute for the paid scout on this instrument, and the limit is throughput and not generation length.** It serializes at 15.1 completion tok/s, so the briefed step 1 alone is 94 hours. At composed@L64, k=6 — n=8, every item finishing on `stop` at 10933 completion tokens a piece and 8.42 measured touches, DEEPER than the 7.71 of composed@128 where glm-5.2 reads 0.575 — DeepSeek V4 scores 1.000 (95% CI [0.68, 1.00]). It is a second ceiling and cannot test the k axis either. The two free models fail the same job from opposite ends: the local Qwen is at informed chance at the lengths it can afford, this one is at the ceiling at the deepest cell it has been run on.

## Why k cannot buy difficulty at fixed L — the task's own geometry

The composed cell's cheapest correct algorithm chases the queried agent's pointer through the swaps that touch it. A swap moves two of the k pointers, so raising k at fixed L divides the chain among more agents. **k divides the chain.** That is a difficulty-REDUCING move on the state leg, not a difficulty-buying one.

> **CORRECTED.** The earlier rendering of this table printed the FORMULA `2 n_swap / k` (`validity.s5_bind_v3_carrier_hops`), which is a uniform expectation. The composed spec's query gates pick the agent the chain is measured on, so the formula UNDERSTATES every cell, increasingly with k. The direction survives and the magnitude does not: across k=6..48 at L=128 the formula divides the chain 8.6x and the streams divide it 5.0x.

Carrier chain, MEASURED off the rendered prompts at n=200 items per cell. `touch` is the number of swaps that NAME the queried agent, which is what the formula estimates; `carry` is the backward carrier walk, the events whose contents the answer actually depends on. `formula` is `2 n_swap / k` at the cell's own measured `n_swap`, printed so the gap is visible rather than asserted:

| L | k=6 | k=12 | k=24 | k=32 | k=48 |
|---|---|---|---|---|---|
| 32 **touch** | **4.97** | **3.18** | **2.56** | **2.35** | **2.23** |
| 32 carry | 4.52 | 2.54 | 1.54 | 1.44 | 1.26 |
| 32 formula | 4.00 | 1.83 | 0.92 | 0.75 | 0.50 |
| 64 **touch** | **8.42** | **4.91** | **3.29** | **2.83** | **2.51** |
| 64 carry | 8.21 | 4.50 | 2.46 | 2.27 | 1.71 |
| 64 formula | 7.33 | 3.67 | 1.75 | 1.38 | 0.92 |
| 128 **touch** | **15.84** | **7.71** | **4.96** | **4.08** | **3.19** |
| 128 carry | 15.63 | 7.80 | 4.36 | 3.54 | 2.65 |
| 128 formula | 15.00 | 7.17 | 3.58 | 2.62 | 1.75 |
| 192 **touch** | **23.42** | **11.39** | **6.58** | **5.19** | **3.94** |
| 192 carry | 22.81 | 11.78 | 6.25 | 5.13 | 3.62 |
| 192 formula | 22.67 | 10.67 | 5.33 | 4.00 | 2.67 |
| 256 **touch** | **30.27** | **14.76** | **8.15** | **6.33** | **5.04** |
| 256 carry | 30.59 | 14.73 | 8.01 | 6.37 | 4.36 |
| 256 formula | 30.33 | 14.17 | 7.08 | 5.38 | 3.54 |

Zero-hop items are 0.000 of every one of the 51 measured cells, so every sub-1-hop entry in the formula row describes no stream that exists: the measured floor over this grid is 2.12 touches (k=64, L=32), against a formula reading of 0.38 there. The two measured rows separate at high k because the carrier MOVES: a swap that names the queried agent may hand the value to an agent no later swap touches.

What k does buy is REAL but is not difficulty: informed chance falls as 1/(k-1), so k=48 reads against 0.0213 where k=12 reads against 0.0909 — 4.3x the measurement resolution, free and model-independent — and the cell holds more live slots, which a scratchpad supplies at the price of tokens rather than of errors.

Holding the chain fixed while raising k means raising L in step, so k's difficulty is bought in L's currency at L's price. The shortest MEASURED cell at each width whose chain reaches 3 touches, and what one n=40 cell costs there on this arm:

| k | shortest measured L with 3+ touches | its measured touch | one n=40 cell there |
|---|---|---|---|
| 6 | 32 | 4.97 | 0.26M completion tokens = 5 h |
| 12 | 32 | 3.18 | 0.26M completion tokens = 5 h |
| 24 | 64 | 3.29 | 0.51M completion tokens = 9 h |
| 32 | 96 | 3.56 | 0.77M completion tokens = 14 h |
| 48 | 128 | 3.19 | 1.02M completion tokens = 18 h |

### The prediction, tested on the grid already in hand

The local Qwen k x L grid is 16 live cells at n=40 (`results/20260802_s5bind_v3_local_kL_sweep.md`). It was set aside this round on the ground that Qwen sits at or below informed chance at three of four lengths, so its flat k axis measured an absence of resolution. That disqualification does not cover the whole grid: **at L=64 Qwen has headroom at every k rung**, and mid-range is where a binomial resolves best.

| k | L=64 chance | L=64 match | x chance | z | measured touch hops |
|---|---|---|---|---|---|
| 6 | 0.2000 | 0.550 | 2.75x | +5.5 | 8.42 |
| 12 | 0.0909 | 0.550 | 6.05x | +10.1 | 4.91 |
| 24 | 0.0435 | 0.700 | 16.10x | +20.4 | 3.29 |
| 32 | 0.0323 | 0.550 | 17.05x | +18.5 | 2.83 |

Every rung is above chance by z=+5.5 or more, so this row has resolution to lose. Across k=6 to k=32 match spans **0.150** against a 95% Wilson half-width of 0.15 at n=40 — flat — while the MEASURED carrier chain falls from 8.42 touches to 2.83.

Read against the baseline k itself moves, the ordering is the wrong way round at 3 of 4 lengths: match-over-chance rises monotonically with k at each of them. The cell does not get harder as it gets wider; it gets easier relative to the guess it is scored against.

Over all 16 live cells: corr(match, log L) = **-0.934**, corr(match, log measured hops) = -0.489, corr(match, log k) = **-0.043**, and corr(log(match / informed chance), log k) = **+0.627**. L is what this model pays for.

### What the k axis is, and why it is not inert for both arms

Priced on the cell's own algorithm at a fixed L=128 — live slots `W = k + m + 1` and steps `S` from `validity.s5_bind_v3_task_cost`, measured chain from the probe:

| k | composed W (live slots) | composed S (steps) | measured touch hops @L128 | informed chance |
|---|---|---|---|---|
| 6 | 13 | 532 | 15.84 | 0.2000 |
| 12 | 25 | 538 | 7.71 | 0.0909 |
| 24 | 49 | 562 | 4.96 | 0.0435 |
| 32 | 65 | 578 | 4.08 | 0.0323 |
| 48 | 97 | 610 | 3.19 | 0.0213 |

Across k=6 to k=48 at fixed L=128 the cell's live-slot requirement rises **7.5x** (13 -> 97), its step count rises 1.15x (532 -> 610), and its measured chain FALLS 5.0x (15.84 -> 3.19).
Along the other axis, k=12 held fixed and L taken 128 -> 512, W does not move at all (25 -> 25) while the measured chain rises 3.8x (7.71 -> 29.30).

**k is a WIDTH axis and L is a DEPTH axis.** That is one fact with two different consequences, and the earlier rendering carried only the first. On the FRONTIER arm the model emits its working as content, so width is bought with tokens rather than with errors and k is inert: it moves the guess baseline and nothing else. On a bounded-state FROM-SCRATCH architecture width is the binding resource — the state has to be held in the recurrent carrier, not in a scratchpad — so k is NOT inert there, and the same axis that buys nothing on the paid arm is the axis the local arm is about.

## What raising k at fixed L does to the composed cell itself

The composed cell earns its name only while maintaining ONE of the two structures is insufficient. So the question is not only whether a model finds the cell harder at higher k, it is whether the cell is still asking the question. What answers it is the strongest policy the class rule ADMITS: a row that holds at most one structure plus a scratch register and pays no more steps than the cell's own algorithm. How often such a row is nonetheless RIGHT is a property of the stream, replayed by `validity.s5_bind_v3_floors` at n=1000 per cell with no model in the loop, and is what the (k, L) choice decides.

> **CORRECTED.** The earlier rendering read this off `one_structure_max` — the larger of `one_structure_P` and `one_structure_B`. `one_structure_B` reads B live and reads the ANSWER out of P, so it holds 1+k+m slots against the task's own 1+k+m: `validity.s5_bind_v3_admits` rejects it at all 51 cells on this grid, and it was the larger of the two at 33 of them. The number below is the max over the rows that ARE admitted (initial_only, last_swap_ref, last_write_1hop, one_structure_P, stated_reference); `one_structure_B` is reported separately below rather than deleted. Two conclusions move with it and are marked where they occur.

This is not a floor and is not read as one: in the scratchpad regime the composed cell has none, and a live-slot bound is exactly what a visible trace defeats. It measures something else — how much of the composed cell's separation from its components a (k, L) choice leaves standing. The stream-blind read (`initial_only`) is 0.0000 in every cell of the grid — the sampler's `q_no_surface` gate gives it no items — so what follows is not that shortcut returning under another name.

Strongest ADMITTED row as a multiple of that cell's own informed chance 1/(k-1), with the row that sets it:

| L | k=6 | k=12 | k=24 | k=32 | k=48 | k=64 |
|---|---|---|---|---|---|---|
| 32 | 1.04x (one_structure_P) | **3.32x (one_structure_P)** | **13.89x (one_structure_P)** | **21.73x (one_structure_P)** | **37.88x (one_structure_P)** | **54.62x (one_structure_P)** |
| 64 | 0.96x (one_structure_P) | 1.10x (one_structure_P) | **5.80x (one_structure_P)** | **13.58x (one_structure_P)** | **26.93x (one_structure_P)** | **42.15x (one_structure_P)** |
| 96 | 0.97x (last_write_1hop) | 1.06x (stated_reference) | **2.60x (one_structure_P)** | **6.08x (one_structure_P)** | **17.77x (one_structure_P)** | **32.26x (one_structure_P)** |
| 128 | 0.81x (last_write_1hop) | 0.99x (last_write_1hop) | 1.08x (one_structure_P) | **2.54x (one_structure_P)** | **10.43x (one_structure_P)** | **24.00x (one_structure_P)** |
| 192 | 0.99x (last_write_1hop) | 1.02x (last_write_1hop) | 1.13x (last_write_1hop) | 1.09x (stated_reference) | **4.32x (one_structure_P)** | **9.95x (one_structure_P)** |
| 256 | 1.01x (one_structure_P) | 1.02x (stated_reference) | 1.15x (one_structure_P) | 0.99x (one_structure_P) | **1.60x (stated_reference)** | **4.54x (one_structure_P)** |
| 384 | — | 0.91x (last_write_1hop) | 1.10x (one_structure_P) | 1.09x (one_structure_P) | 1.13x (one_structure_P) | **2.02x (one_structure_P)** |
| 512 | — | 0.97x (last_write_1hop) | 1.17x (last_write_1hop) | 1.15x (last_write_1hop) | 1.18x (one_structure_P) | **1.51x (last_write_1hop)** |
| 768 | — | 1.13x (last_write_1hop) | 1.10x (stated_reference) | 1.18x (last_write_1hop) | **1.32x (last_write_1hop)** | **1.26x (one_structure_P)** |

Bold is over 1.25x — a chosen line, not a measured one; the raw numbers are printed so a different line can be drawn. Above it a cheaper solver is beating the guess baseline the cell is scored against. Raising k at fixed L walks every row rightwards into that region: at L=128 the read goes 0.0900 (0.99x) at k=12 to 0.2220 (10.43x) at k=48, and at L=32/k=48 it reaches 0.8060 (37.88x). k does not make the composed cell harder; past a point it makes it a component cell with more agents in it.

### The full-width one-structure row, reported separately

`one_structure_B` carries B through the gives, resolves every reference against the stated P0, and reads the answer out of P — so it holds both structures and is not a cheaper algorithm. It is not admitted and sets nothing; what it measures is how often tracking the OTHER structure alone lands on the answer, which is a different fact and moves the other way at long L:

| L | k=6 | k=12 | k=24 | k=32 | k=48 | k=64 |
|---|---|---|---|---|---|---|
| 32 | 1.09x | 4.41x | 15.55x | 23.00x | 39.62x | 56.26x |
| 64 | 0.83x | 1.63x | 8.33x | 14.82x | 31.02x | 47.19x |
| 96 | 0.80x | 0.90x | 4.51x | 9.27x | 22.65x | 38.30x |
| 128 | 0.77x | 0.85x | 2.60x | 5.74x | 16.87x | 31.25x |
| 192 | 0.78x | 0.75x | 1.33x | 2.48x | 9.63x | 19.47x |
| 256 | 0.88x | 1.06x | 0.97x | 1.09x | 4.32x | 11.72x |
| 384 | — | 0.96x | 0.97x | 0.74x | 1.32x | 4.47x |
| 512 | — | 0.82x | 0.94x | 0.99x | 1.13x | 1.20x |
| 768 | — | 0.86x | 1.01x | 0.96x | 0.85x | 0.69x |

### The shape of the admissible region

L/k is most of the story and not all of it. L/k is how many times the stream touches any one agent or object, so as it falls the two structures stop moving under each other and the RAW read falls with it. But the baseline the cell is scored against falls as 1/(k-1) at the same time, and it falls faster — so at matched L/k the read gets WORSE as a multiple of chance the wider the cell is:

| L/k | cells (k, L) | admitted read | x chance |
|---|---|---|---|
| 0.50 | (64, 32) | 0.8670 | 54.62x |
| 0.67 | (48, 32) | 0.8060 | 37.88x |
| 1.00 | (32, 32), (64, 64) | 0.7010, 0.6690 | 21.73x, 42.15x |
| 1.33 | (24, 32), (48, 64) | 0.6040, 0.5730 | 13.89x, 26.93x |
| 1.50 | (64, 96) | 0.5120 | 32.26x |
| 2.00 | (32, 64), (48, 96), (64, 128) | 0.4380, 0.3780, 0.3810 | 13.58x, 17.77x, 24.00x |
| 2.67 | (12, 32), (24, 64), (48, 128) | 0.3020, 0.2520, 0.2220 | 3.32x, 5.80x, 10.43x |
| 3.00 | (32, 96), (64, 192) | 0.1960, 0.1580 | 6.08x, 9.95x |
| 4.00 | (24, 96), (32, 128), (48, 192), (64, 256) | 0.1130, 0.0820, 0.0920, 0.0720 | 2.60x, 2.54x, 4.32x, 4.54x |
| 5.33 | (6, 32), (12, 64), (24, 128), (48, 256) | 0.2080, 0.1000, 0.0470, 0.0340 | 1.04x, 1.10x, 1.08x, 1.60x |
| 6.00 | (32, 192), (64, 384) | 0.0350, 0.0320 | 1.09x, 2.02x |
| 8.00 | (12, 96), (24, 192), (32, 256), (48, 384), (64, 512) | 0.0960, 0.0490, 0.0320, 0.0240, 0.0240 | 1.06x, 1.13x, 0.99x, 1.13x, 1.51x |
| 10.67 | (6, 64), (12, 128), (24, 256), (48, 512) | 0.1930, 0.0900, 0.0500, 0.0250 | 0.96x, 0.99x, 1.15x, 1.18x |
| 12.00 | (32, 384), (64, 768) | 0.0350, 0.0200 | 1.09x, 1.26x |
| 16.00 | (6, 96), (12, 192), (24, 384), (32, 512), (48, 768) | 0.1940, 0.0930, 0.0480, 0.0370, 0.0280 | 0.97x, 1.02x, 1.10x, 1.15x, 1.32x |
| 21.33 | (6, 128), (12, 256), (24, 512) | 0.1630, 0.0930, 0.0510 | 0.81x, 1.02x, 1.17x |
| 24.00 | (32, 768) | 0.0380 | 1.18x |
| 32.00 | (6, 192), (12, 384), (24, 768) | 0.1990, 0.0830, 0.0480 | 0.99x, 0.91x, 1.10x |
| 42.67 | (6, 256), (12, 512) | 0.2020, 0.0880 | 1.01x, 0.97x |
| 64.00 | (12, 768) | 0.1030 | 1.13x |

The smallest L/k at which each width still reads at chance, measured:

| k | smallest admissible L/k | i.e. L at least |
|---|---|---|
| 6 | 5.3 | 32 |
| 12 | 5.3 | 64 |
| 24 | 5.3 | 128 |
| 32 | 6.0 | 192 |
| 48 | 8.0 | 384 |

The requirement rises with k, from L/k >= 5.3 at k=6 to L/k >= 8.0 at k=48. Of the 51 cells on this grid, 27 are admissible. k=64 is admissible at no length measured here, up to L=768.

> **CORRECTED.** The earlier rendering called this **a narrowing ray**, on a requirement that ran L/k >= 5.3 at k=6 to L/k >= 10.7 by k=24 with 23 of 51 cells admissible. On the admitted rows the requirement is flat at L/k >= 5.3 for k=6, 12 and 24 and rises only at the top two widths, and 27 of 51 cells are admissible. The region still narrows; it narrows about half as fast, and the consequence is that a given L buys a wider cell than was priced.

| L | largest k that stays within 1.25x | its read | next k up | its read | earlier rendering's k |
|---|---|---|---|---|---|
| 32 | 6 | 1.04x | 12 | 3.32x | 6 |
| 64 | 12 | 1.10x | 24 | 5.80x | 6 |
| 96 | 12 | 1.06x | 24 | 2.60x | 12 |
| 128 | 24 | 1.08x | 32 | 2.54x | 12 |
| 192 | 32 | 1.09x | 48 | 4.32x | 12 |
| 256 | 32 | 0.99x | 48 | 1.60x | 32 |
| 384 | 48 | 1.13x | 64 | 2.02x | 32 |
| 512 | 48 | 1.18x | 64 | 1.51x | 48 |
| 768 | 32 | 1.18x | 48 | 1.32x | 48 |

The column runs non-monotonically in k at the long end, which is why the whole row is printed above rather than only its argmax: at L=768 the reads are k=12: 1.13x, k=24: 1.10x, k=32: 1.18x, k=48: 1.32x, k=64: 1.26x, so the widest cell inside the line is not the widest cell with the lowest read.

### A second thing k does to the stream

A row that moves even further is worth stating separately: `window_90`, which replays the task's own algorithm but SKIPS THE FIRST TENTH of the events. It holds both structures and so is not a cheaper solver — `validity.s5_bind_v3_admits` rejects it for exactly that reason — but it is 0.90x the task's steps, and what it measures is how much of the stream carries no information about the answer.

| L | k=6 | k=12 | k=24 | k=32 | k=48 | k=64 |
|---|---|---|---|---|---|---|
| 32 | 1.91x | 6.36x | 18.68x | 26.94x | 43.48x | 60.35x |
| 64 | 1.18x | 2.16x | 11.64x | 20.24x | 37.69x | 54.18x |
| 96 | 0.91x | 1.07x | 5.84x | 12.56x | 28.86x | 45.23x |
| 128 | 0.91x | 1.04x | 3.01x | 7.59x | 21.71x | 36.48x |
| 192 | 0.89x | 1.09x | 1.52x | 2.29x | 10.90x | 25.33x |
| 256 | 0.92x | 0.98x | 1.13x | 1.21x | 4.32x | 12.54x |
| 384 | — | 0.87x | 1.06x | 0.99x | 1.13x | 4.28x |
| 512 | — | 1.08x | 1.08x | 1.24x | 1.27x | 1.70x |
| 768 | — | 1.00x | 0.94x | 0.96x | 0.89x | 1.32x |

At k=64, L=32 a solver that never reads the first tenth of the stream answers correctly 0.958 of the time against an informed chance of 0.0159 — 60x. The events are still there; at that width they have stopped mattering. This is the same fact as the falling measured chain seen from the readout side, and it is why raising k cannot substitute for raising L: L is what puts events between the query and the initial map, and k takes them back out.

### What that costs a live reading

The margin a measured score holds OVER the admitted read is what the composed cell is buying. On the local Qwen grid at n=40 that margin collapses along k while raw match barely moves — the flatness of the k axis is not a null, it is the cell handing its own separation away:

| L | k | match (n=40) | admitted read | margin | x chance |
|---|---|---|---|---|---|
| 64 | 6 | 0.550 | 0.1930 | **+0.357** | 2.75x |
| 64 | 12 | 0.550 | 0.1000 | **+0.450** | 6.05x |
| 64 | 24 | 0.700 | 0.2520 | **+0.448** | 16.10x |
| 64 | 32 | 0.550 | 0.4380 | **+0.112** | 17.05x |
| 128 | 6 | 0.250 | 0.1630 | **+0.087** | 1.25x |
| 128 | 12 | 0.250 | 0.0900 | **+0.160** | 2.75x |
| 128 | 24 | 0.200 | 0.0470 | **+0.153** | 4.60x |
| 128 | 32 | 0.225 | 0.0820 | **+0.143** | 6.98x |
| 192 | 6 | 0.125 | 0.1990 | **-0.074** | 0.62x |
| 192 | 12 | 0.075 | 0.0930 | **-0.018** | 0.82x |
| 192 | 24 | 0.075 | 0.0490 | **+0.026** | 1.73x |
| 192 | 32 | 0.125 | 0.0350 | **+0.090** | 3.88x |
| 256 | 6 | 0.200 | 0.2020 | **-0.002** | 1.00x |
| 256 | 12 | 0.075 | 0.0930 | **-0.018** | 0.82x |
| 256 | 24 | 0.100 | 0.0500 | **+0.050** | 2.30x |
| 256 | 32 | 0.075 | 0.0320 | **+0.043** | 2.33x |

L=64 is the row where this model has headroom at every rung, so it is the row that can lose something. Across it match goes 0.550 (k=6) to 0.550 (k=32) — flat, which is what the k axis was reported as buying nothing — while the margin over the admitted class runs k=6: +0.357, k=12: +0.450, k=24: +0.448, k=32: +0.112. The axis is not inert: what it moves is the cell's discriminating power, and it returns a flat score while doing it.

The region is bounded from the other side too, by the sampler rather than by a solver: the composed spec's `q_no_surface` gate could not fill the n=1000 draw at 3 of the grid's cells, every one of them k=6 at L>=384. The narrow-and-long corner is not available, so the ray has a floor in k as well as a ceiling.

The shipped operating point is k=12 at L=128 — L/k = 10.7, read 0.99x. It sits INSIDE the frontier and not on it: the next k rung up at L=128, k=24, reads 1.08x and is admissible too, so the registered cell is one rung narrower than that length allows.

> **CORRECTED.** The earlier rendering read "the shipped operating point sits ON the edge" off `one_structure_B`, which put k=24 at L=128 at 2.60x. On the admitted rows k=24 at L=128 reads 1.08x, so there is one k rung of headroom at the shipped length that the earlier reading priced away.

## Where the next paid point goes, and what it costs

STOP_CEILING asked for a cell a strong model is off the ceiling on. The rule's own remedy was "raise k or L". The k half is ruled out above on two independent grounds — k divides the measured chain, and past the admissibility ray it hands the cell to a cheaper solver that need not hold both structures — so the remaining direction is L, with k raised only as far as the ray allows and only for the resolution it buys.

`openai/gpt-5.5` at the registered k=12, measured: match 1.000 at L=128 and 0.975 at L=256, on 7740 and 13457 completion tokens per item — 45 completion and 17 prompt tokens per event, and $0.244 and $0.427 an item. Both cells are at the ceiling, and the deeper of them carries 14.76 measured touches. Extending that per-event rate along the admissible ray:

| L | widest admissible k | its admitted read | informed chance | measured touch hops | est. completion tok/item | est. $/item | est. $ at n=40 | earlier rendering's k (hops) |
|---|---|---|---|---|---|---|---|---|
| 256 | 32 | 0.99x | 0.0323 | 6.3 | 13457 | $0.43 | $17 | 32 (6.3) |
| 384 | 48 | 1.13x | 0.0213 | 6.5 | 19175 | $0.61 | $24 | 32 (9.1) |
| 512 | 48 | 1.18x | 0.0213 | 8.0 | 24892 | $0.79 | $32 | 48 (8.0) |
| 768 | 32 | 1.18x | 0.0323 | 16.2 | 36328 | $1.16 | $46 | 48 (11.4) |

> **CORRECTED.** The `widest admissible k` column moved when the ray was recomputed over admitted rows, and it changes which cell to buy at a given price. At L=768 the earlier rendering priced k=48, whose measured chain is 11.36 touches; the admitted read permits k=32 at 1.18x, whose chain is 16.18 — 1.42x the depth for the same money — and k=24 at 1.10x carries 21.50, 1.89x the depth at a 2.2x coarser guess baseline.

The estimate is linear in L at this model's own measured per-event rate, which is what it spends AT THE CEILING; a model that is actually working will spend more, so these are lower bounds on the completion side. It ignores k, which on the local grid LOWERED completion tokens at fixed L (k=6 to k=32 at L=128: 18,654 to 11,745 per item), so the widening is if anything cheaper than priced. These rows and the probe priced at the end of this report are the only extrapolations here and are marked as such; everything else is a measurement.

Widening along the ray is bought for RESOLUTION and not for difficulty: at L=512 the widest admissible cell is k=48, whose informed chance is 0.0213 against 0.0909 at the shipped point — 4.3x the measurement resolution — but its measured chain is 8.0 touches against 29.3 for k=12 at the same length. The difficulty comes from L. k comes along to keep the guess baseline low enough that the difficulty is readable.

## What this arm can and cannot buy

Priced at 200 completion tokens per event and 15.1 completion tokens per second, both measured on this arm:

| step | cells | completion tokens | serialized wall clock |
|---|---|---|---|
| 1. place the model | the four registered k=12 cells at n=40 | 5.12M | **94 h** |
| 2. k in {6,12,24,32,48} at n=40, L=64 | 5 composed cells | 2.56M | **47 h** |
| 2. k in {6,12,24,32,48} at n=40, L=128 | 5 composed cells | 5.12M | **94 h** |
| 2. k in {6,12,24,32,48} at n=40, L=256 | 5 composed cells | 10.24M | **188 h** |
| 3. L in {64,128,192,256} at n=40, one k | 4 composed cells | 5.12M | **94 h** |

One n=40 composed cell at L=128 is 1.02M completion tokens, which this endpoint delivers in 19 hours. The briefed round is 282+ hours on a server that runs one request at a time. It was not run. What was run is priced against the same numbers and stated at its own n.

| cell | k | L | n | completion tokens | wall clock |
|---|---|---|---|---|---|
| composed | 6 | 32 | 20 | 125021 | 2.3 h |
| composed | 6 | 64 | 8 | 87466 | 1.7 h |
| state | 6 | 12 | 20 | 45737 | 0.8 h |

### The endpoint, measured

The window is not what bounds this grid; the WALL CLOCK is. ds4-server holds one KV session and serializes — at 1/2/4/8 concurrent calls throughput is flat at 16.3-16.4 completion tok/s while per-call latency scales linearly, and the server's own log carries a single generation stream for four concurrent requests. So a cell's duration is its completion tokens divided by one number.

| probe | cell | k | L | workers | budget | completion tokens | wall | completion tok/s | finish |
|---|---|---|---|---|---|---|---|---|---|
| w1_state | state | 12 | 43 | 1 | 16384 | 5411 | 344s | 15.72 | `['stop']` |

Across every scored cell in this round the endpoint delivered **15.1 completion tokens per second** (258224 completion tokens in 4.7 hours of wall clock). That is the number every cell above is priced at.

### What bounds an item on this arm: throughput, not generation length

> **CORRECTED.** The earlier rendering said "the endpoint does not return a generation of ~10.5k tokens at all" and used it, alongside the wall clock, as one of two independent reasons the band cells could not be run. It is disproved by a cell in this round's own history.

composed@L64, k=6, n=8: every item returned, `{'stop': 8}`, 0.00 truncated, 0.00 empty, 0 API errors, per-item completion tokens 7,775-13,389 (median 10,854). 6 of 8 items are at or over 10,000 completion tokens and 5 are at or over the 10,482 the earlier reading called unreturnable. The whole cell took 1.7 h at 14.4 completion tok/s.

The abandoned k=12/L=64 item is recorded in this repository's own history as having completed once at 10,482 tokens in 698.8 s with `finish=stop` and no re-issue (commit `ce1cb35`). The stall is therefore a state the server can enter and not a length the server refuses, and the only standing limit on this arm is throughput: a cell's duration is its completion tokens divided by one number, and that is what the pricing below rests on.

The one path test taken this round, kept because it is VOID and a reader should not re-derive it:

| path | k | L | completion tokens | wall | result |
|---|---|---|---|---|---|
| tunnel | 6 | 32 | — | 223s | **APIConnectionError** — VOID as a path test: the server was stopped mid-queue by this session's own `serve_steed_model.py down`, so the connection error is this shutdown and not the transport |

The stall itself is still unexplained and is a validity risk wherever it recurs: for the item that would not come back the server logged `gen=10482 finish=stop` four times against one prompt, byte-identical, each followed by `live kv cache miss ... reason=token-mismatch` and an immediate re-`prompt start` on the same cached prompt, and went on doing so after the client process was killed. `serve_steed_model.py down`/`up` clears it. A cell that hits it records empty predictions, which score wrong, so the empty-rate column is the gate that catches it — and it read 0.00 on every cell in this round.

## Validity first — every cell's truncation and empty rate

| cell | k | L | n | budget | finish=length | empty | finish reasons | api errors | VOID |
|---|---|---|---|---|---|---|---|---|---|
| composed | 6 | 32 | 20 | 12288 | 0.00 | 0.00 | `{'stop': 20}` | 0 (+0 finish=error) | — |
| composed | 6 | 64 | 8 | 32768 | 0.00 | 0.00 | `{'stop': 8}` | 0 (+0 finish=error) | — |
| state | 6 | 12 | 20 | 8192 | 0.00 | 0.00 | `{'stop': 20}` | 0 (+0 finish=error) | — |

A cell over 10% finish=length or empty is VOID and enters no comparison until it is re-run at a raised budget: a truncated call is scored wrong, so a truncated cell reads as a floor. The published s5 L64 cliff was a 16-token budget read as a capability, which is why the order is validity first and numbers second.

## 1. Where the model sits against the scouted band

The scout's numbers are the registered k=12 spec at n=40 on the answer read — the same cells and the same read as the rows below. The `measured touch` row is the chain each band cell carries, read off its own rendered prompts, so a cell run at another (k, L) can be compared on depth instead of on its label.

| model | composed@128 | composed@256 | state@85 | bind@171 |
|---|---|---|---|---|
| openai/gpt-5.5 | 1.000 | 0.975 | 1.000 | 1.000 |
| nvidia/nemotron-3-ultra-550b-a55b | 0.750 | 0.500 | 0.875 | 1.000 |
| z-ai/glm-5.2 | 0.575 | 0.450 | 0.950 | 0.850 |
| **steed/deepseek-v4-flash** | — | — | — | — |
| *measured touch* | *7.71* | *14.76* | *—* | *—* |

**composed@128, composed@256, state@85, bind@171 were not run on this arm, so this model is not literally placed in the band.** The limit is one and not two: the four band cells at n=40 are 94 hours of serialized generation on this endpoint. The cells that were run are off the band's (k, L) axis and are compared to it on measured chain depth instead.

> **CORRECTED.** The earlier rendering gave a second, independent reason — "the endpoint does not return a generation of ~10.5k tokens at all" — and the placement control below disproves it. Wall clock is the only limit.

**What the round does establish is that this is a SECOND CEILING**, and it rests on the placement control rather than on the shallowest cell. composed@L64, k=6: match 1.000 at n=8, `{'stop': 8}`, 0.00 truncated, 0.00 empty, 95% CI [0.68, 1.00], z=+5.66 over informed chance 0.2000. That cell carries 8.42 measured touches — DEEPER than composed@128's 7.71, where glm-5.2 reads 0.575 and nemotron 0.750.
Against the local Qwen at the IDENTICAL cell (0.550, n=40), Fisher exact p=0.018: the two free arms are separated at the one cell both have been run on.

> **CORRECTED.** The earlier rendering asserted the second ceiling from composed@L32, k=6 (1.000 at n=20), which carries 4.97 measured touches against a band at 7.71 and 14.76. A cell that shallow cannot rank a model on that band; the L=64 control can, and it is what the claim now rests on.

So this arm cannot test the k axis from below any more than the local Qwen can from above: Qwen is at informed chance at the lengths it can afford, this model is at the ceiling at the deepest cell it has been run on, and the depth where it would come off the ceiling is priced in hours below.

## 2. The k axis — match against informed chance 1/(k-1), and its token price

| k | L | chance 1/(k-1) | n | match | 95% CI | x chance | z | measured touch | admitted read | margin | prompt tok/item | completion tok/item | wall/item |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 6 | 32 | 0.2000 | 20 | 1.000 | [0.84,1.00] | 5.00x | +8.9 | 4.97 | 0.2080 (1.04x) | **+0.792** | 702 | 6251 | 414s |
| 6 | 64 | 0.2000 | 8 | 1.000 | [0.68,1.00] | 5.00x | +5.7 | 8.42 | 0.1930 (0.96x) | **+0.807** | 1244 | 10933 | 758s |

`admitted read` is that cell's strongest admitted policy from the section above, and `margin` is what the measured score holds over it. A cell whose margin is not clearly positive is not reading composition, whatever its match says.

## 3. The L axis, at the same k and the same protocol

At k=6, L in [32, 64]: match spans **0.000**, over a measured chain that goes 4.97 touches to 8.42 — 1.7x the depth for no movement in the score. An extra EVENT costs +146 completion tokens and +17 prompt tokens per item.

## Component cells, against floors recomputed from the exact scored items

| cell | k | L | partner of composed@L | n | match | floor (scored) | floor (disjoint) | operative | basis | x floor |
|---|---|---|---|---|---|---|---|---|---|---|
| state | 6 | 12 | 32 | 20 | 1.000 | 0.2000 | 0.2000 | **0.2000** | measured | 5.00x |

Both floors are printed because they measure different failure modes: the max over admitted rows carries an upward selection bias at small n, and the house rule is that a floor is recomputed from the items a score is actually read against. The larger is operative. A component floor's admitted rows are depth <= 1 and cost under the cell's own algorithm's per-item minimum, so a scratchpad does not void them — a pad substitutes for registers, not for chaining.

## 4. The operating point on this model: off the ceiling, components still solved

| k | L | composed | state partner (match) | bind partner (match) | off the ceiling? | components solved? |
|---|---|---|---|---|---|---|
| 6 | 32 | 1.000 | 12 (1.000) | 20 (—) | NO (at the ceiling) | not measured |
| 6 | 64 | 1.000 | 22 (—) | 42 (—) | NO (at the ceiling) | not measured |

No cell measured on this arm satisfies both conditions. The question is asked of a STRONG model in any case, and the scouted three were read at k=12, L=128 and L=256 — lengths this arm cannot run — so the operating point for a redesign is the one priced above from the scout's own usage, not one this model could be walked to.

## The one paid probe the evidence supports — priced for decision, NOT issued

**The cell: composed, k=12, L=512, n=12, on `openai/gpt-5.5` and `z-ai/glm-5.2`.** No call was made. This section exists so the decision can be taken on numbers rather than on a plan.

**Why this cell and not another.** It is the deepest admitted cell at the SHIPPED width: 29.30 measured touches against composed@256's 14.76 and composed@128's 7.71 — 3.8x the depth of the shallower cell gpt-5.5 already reads 1.000 on, with the strongest admitted policy at 0.97x informed chance (0.0909), so the cell is still asking its own question. Holding k at 12 keeps the guess baseline and the component partners identical to the scouted band, so the only thing that moves is depth.

**The estimate, and its basis.** Linear extrapolation in L of each model's OWN measured per-item usage at k=12, L=128 and L=256 (`results/s5bind_v3_scout/`, VOID cells excluded), at the registry's prices. It is an EXTRAPOLATION and is the only number in this section that is not measured:

| model | measured L=128 | measured L=256 | ctok/event | est. ctok/item @512 | est. $/item | est. $ at n=12 |
|---|---|---|---|---|---|---|
| `openai/gpt-5.5` | 1.000 @ 7740 ctok | 0.975 @ 13457 ctok | 45 | 24892 | $0.79 | $9.50 |
| `z-ai/glm-5.2` | 0.575 @ 13485 ctok | 0.450 @ 24423 ctok | 85 | 46300 | $0.15 | $1.77 |

**Total: $11.27** at the ceiling rate. A model that is actually working spends more, so the completion side is a lower bound; the cost guard should be set at 2x, i.e. $23, and a budget of 65,536 completion tokens for gpt-5.5 and 131,072 for glm-5.2 (both above 2x their estimated mean, since a budget under the trace length scores the cell as a floor).

**The pre-registered decision rule, in force before the probe is issued.** The comparison is Fisher exact against each model's OWN measured L=256 cell at n=40, which is the only reading that answers "did the depth move this model" rather than "is this model good". The whole decision boundary is printed, so the rule is a threshold and not a judgement made after the fact:

| `openai/gpt-5.5` at n=12 | 95% Wilson | Fisher p vs its own 39/40 at L=256 | verdict |
|---|---|---|---|
| 12/12 = 1.000 | [0.76, 1.00] | 1.000 | **STOP — the ceiling is not an L problem** |
| 11/12 = 0.917 | [0.65, 0.99] | 0.412 | **STOP — the ceiling is not an L problem** |
| 10/12 = 0.833 | [0.55, 0.95] | 0.129 | **STOP — the ceiling is not an L problem** |
| 9/12 = 0.750 | [0.47, 0.91] | 0.034 | the L axis is live |
| 8/12 = 0.667 | [0.39, 0.86] | 0.008 | the L axis is live |

- **STOP AND REDESIGN** if `openai/gpt-5.5` reads 10/12 or better. A 3.8x increase in measured depth over the cell it already reads 1.000 on would then have moved it by less than this test resolves, and **no larger-L purchase is warranted**: the ceiling is not an L problem. Do not re-budget — the last two rounds were lost to exactly that move.
- **THE L AXIS IS LIVE** if it reads 9/12 or worse. The next purchase is then this same cell at n=40 plus its two work-matched component partners (state@172, bind@340), which is what a placement needs and what n=12 cannot give.
- `z-ai/glm-5.2` is NOT a second test of the same thing. Its job is to say the cell is answerable and not VOID at this length. Against its own 18/40 at L=256, n=12 resolves a drop only at 1/12 or below (p<0.05); anything above that is uninformative about it, and the report should say so rather than read the middle of that range.
- Either way the result is read against the admitted class at this cell (0.97x informed chance) and against informed chance 0.0909, never against 1/k. VALIDITY FIRST as everywhere: over 10% finish=length or empty and the cell is VOID and decides nothing.

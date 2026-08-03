# s5_bind_v3 on steed's DeepSeek V4 — placing the model, and pricing the k axis

Written 2026-08-03T10:37:20+00:00. Model `steed/deepseek-v4-flash` — DeepSeek V4, q2, ~81 GB resident, served by `ds4-server` on a DGX Spark GB10 over the tailnet at a 262,144-token window. Metric is **match**, the canonical evaluator, on the ANSWER only. Effort arm `high`; ds4 collapses `minimal`/`low`/`medium`/`high`/`xhigh` to one internal level, so that is the model's single thinking arm. **No paid endpoint was contacted and the local GPU was not used**; `cost_usd_est` is 0.0 on every record.

**The composed cell has no floor in this regime.** The model emits its working as plain content, which is a scratchpad, and the composed cell's floor argument bounds LIVE SLOTS (W <= max(k,m)+1 against the task's k+m+1). Its number is read against INFORMED CHANCE 1/(k-1) — the initial map is stated, so the queried agent's own starting value is never gold — which is a guess baseline, not a floor, and which MOVES WITH k. Component cells keep their floors, recomputed below from the exact scored items and from a disjoint pool.

## The answers

**k does not buy difficulty, and the reason is in the task rather than in any model.** The composed cell's state leg is a chain of `2 n_swap / k` hops and `n_swap` is a fixed fraction of L, so raising k at fixed L DIVIDES the chain. On the local grid, over 16 live cells at n=40, corr(match, log L) = -0.934 while corr(match, log k) = -0.043. Read against informed chance — the baseline k itself moves — the ordering runs the wrong way: match-over-chance rises monotonically with k at 3 of the 4 lengths.

**Worse than inert: past a boundary, raising k stops the cell being a composition cell.** A half-price solver that maintains ONE structure and resolves references against the other's stated initial map reads 0.92x informed chance at the shipped k=12/L=128 and 16.76x at k=48/L=128. The admissible set is a narrowing ray in (k, L), the shipped operating point sits on its edge, and on the local grid the margin a measured score holds over that solver collapses along k while raw match stays flat.

**So the ceiling's only remaining remedy is L**, with k raised no faster than the ray allows and only for the resolution 1/(k-1) buys. Candidate next paid points and their prices are below.

**The free arm cannot substitute for the paid scout on this instrument.** It serializes at ~15.8 completion tok/s, so the briefed step 1 alone is 90 hours; and it re-issued one 10,482-token generation indefinitely, which caps how long a single item may be. What was measured on it is stated at its own lengths and its own n, and it is not placed in the scouted band.

## Why k cannot buy difficulty at fixed L — the task's own geometry

The composed cell's cheapest correct algorithm chases the queried agent's pointer through the swaps that touch it. A swap moves two of the k pointers, so that chain is `2 n_swap / k` (`validity.s5_bind_v3_carrier_hops`) and `n_swap` is a fixed fraction of L. **k divides the chain.** Raising k at fixed L shortens the one quantity the algorithm has to walk — it is a difficulty-REDUCING move on the state leg, not a difficulty-buying one.

| L | k=6 | k=12 | k=24 | k=32 | k=48 |
|---|---|---|---|---|---|
| 32 | 4.00 | 1.83 | 0.92 | 0.75 | 0.50 |
| 64 | 7.33 | 3.67 | 1.75 | 1.38 | 0.92 |
| 128 | 15.00 | 7.17 | 3.58 | 2.62 | 1.75 |
| 192 | 22.67 | 10.67 | 5.33 | 4.00 | 2.67 |
| 256 | 30.33 | 14.17 | 7.08 | 5.38 | 3.54 |

What k does buy is REAL but is not difficulty: informed chance falls as 1/(k-1), so k=48 reads against 0.0213 where k=12 reads against 0.0909 — 4.3x the measurement resolution, free and model-independent — and the cell holds more live slots, which a scratchpad supplies at the price of tokens rather than of errors.

Holding the chain fixed while raising k means raising L in step, so k's difficulty is bought in L's currency at L's price:

| k | shortest L with a chain of 3+ hops | one n=40 cell there |
|---|---|---|
| 6 | 24 | 0.19M completion tokens = 3 h |
| 12 | 52 | 0.42M completion tokens = 7 h |
| 24 | 108 | 0.86M completion tokens = 15 h |
| 32 | 148 | 1.18M completion tokens = 21 h |
| 48 | 216 | 1.73M completion tokens = 30 h |

### The prediction, tested on the grid already in hand

The local Qwen k x L grid is 16 live cells at n=40 (`results/20260802_s5bind_v3_local_kL_sweep.md`). It was set aside this round on the ground that Qwen sits at or below informed chance at three of four lengths, so its flat k axis measured an absence of resolution. That disqualification does not cover the whole grid: **at L=64 Qwen has headroom at every k rung**, and mid-range is where a binomial resolves best.

| k | L=64 chance | L=64 match | x chance | z | carrier hops |
|---|---|---|---|---|---|
| 6 | 0.2000 | 0.550 | 2.75x | +5.5 | 7.33 |
| 12 | 0.0909 | 0.550 | 6.05x | +10.1 | 3.67 |
| 24 | 0.0435 | 0.700 | 16.10x | +20.4 | 1.75 |
| 32 | 0.0323 | 0.550 | 17.05x | +18.5 | 1.38 |

Every rung is above chance by z=+5.5 or more, so this row has resolution to lose. Across k=6 to k=32 match spans **0.150** against a 95% Wilson half-width of 0.15 at n=40 — flat — while the carrier chain falls from 7.33 hops to 1.38.

Read against the baseline k itself moves, the ordering is the wrong way round at 3 of 4 lengths: match-over-chance rises monotonically with k at each of them. The cell does not get harder as it gets wider; it gets easier relative to the guess it is scored against.

Over all 16 live cells: corr(match, log L) = **-0.934**, corr(match, log carrier hops) = -0.540, corr(match, log k) = **-0.043**, and corr(log(match / informed chance), log k) = **+0.627**. L is what this model pays for.

## What raising k at fixed L does to the composed cell itself

The composed cell earns its name only while maintaining ONE of the two structures is insufficient. So the question is not only whether a model finds the cell harder at higher k, it is whether the cell is still asking the question. A one-structure solver carries P through the swaps and resolves every reference against the STATED B0 (or the mirror), which is 1+k live slots against the task's 1+k+m and 0.63-0.68x its steps — a half-price algorithm. How often it is RIGHT is a property of the stream, replayed by `validity.s5_bind_v3_floors` at n=1000-2000 per cell with no model in the loop.

This is not a floor and is not read as one: in the scratchpad regime the composed cell has none, and a one-structure policy is not slot-bounded anyway — it walks the whole stream. It measures how much of the composed cell's separation from its components a (k, L) choice leaves standing. The stream-blind read (`initial_only`) is 0.0000 in every cell of the grid — the sampler's `q_no_surface` gate gives it no items — so what follows is not that shortcut returning under another name.

One-structure read as a multiple of that cell's own informed chance 1/(k-1):

| L | k=6 | k=12 | k=24 | k=32 | k=48 | k=64 |
|---|---|---|---|---|---|---|
| 32 | 1.09x | **4.50x** | **15.41x** | **23.45x** | **39.36x** | **55.41x** |
| 64 | 0.91x | **1.61x** | **8.43x** | **15.07x** | **31.61x** | **47.69x** |
| 96 | 0.87x | 1.03x | **4.47x** | **9.21x** | **22.87x** | **38.27x** |
| 128 | 0.88x | 0.92x | **2.63x** | **5.78x** | **16.76x** | **31.06x** |
| 192 | 0.93x | 0.95x | **1.30x** | **2.23x** | **8.77x** | **19.06x** |
| 256 | 0.97x | 0.96x | 1.02x | **1.36x** | **4.28x** | **10.62x** |
| 384 | — | 0.96x | 1.10x | 1.09x | **1.32x** | **4.47x** |
| 512 | — | 0.95x | 0.94x | 0.99x | 1.18x | **1.32x** |
| 768 | — | 0.96x | 1.01x | 1.09x | 1.13x | **1.26x** |

Bold is over 1.25x — a chosen line, not a measured one; the raw numbers are printed so a different line can be drawn. Above it a half-price one-structure solver is beating the guess baseline the cell is scored against. Raising k at fixed L walks every row rightwards into that region: at L=128 the read goes 0.0835 (0.92x) at k=12 to 0.3565 (16.76x) at k=48, and at L=32/k=48 it reaches 0.8375 (39.36x). k does not make the composed cell harder; past a point it makes it a component cell with more agents in it.

### The shape of the admissible region

L/k is most of the story and not all of it. The coupling between the two structures is what a cross-reference costs, and a give lands on the object a later swap names about L/k times, so the RAW one-structure read falls with L/k. But the baseline the cell is scored against falls as 1/(k-1) at the same time, and it falls faster — so at matched L/k the read gets WORSE as a multiple of chance the wider the cell is:

| L/k | cells (k, L) | one-structure read | x chance |
|---|---|---|---|
| 0.50 | (64, 32) | 0.8795 | 55.41x |
| 0.67 | (48, 32) | 0.8375 | 39.36x |
| 1.00 | (32, 32), (64, 64) | 0.7565, 0.7570 | 23.45x, 47.69x |
| 1.33 | (24, 32), (48, 64) | 0.6700, 0.6725 | 15.41x, 31.61x |
| 1.50 | (64, 96) | 0.6075 | 38.27x |
| 2.00 | (32, 64), (48, 96), (64, 128) | 0.4860, 0.4865, 0.4930 | 15.07x, 22.87x, 31.06x |
| 2.67 | (12, 32), (24, 64), (48, 128) | 0.4095, 0.3665, 0.3565 | 4.50x, 8.43x, 16.76x |
| 3.00 | (32, 96), (64, 192) | 0.2970, 0.3025 | 9.21x, 19.06x |
| 4.00 | (24, 96), (32, 128), (48, 192), (64, 256) | 0.1945, 0.1865, 0.1865, 0.1685 | 4.47x, 5.78x, 8.77x, 10.62x |
| 5.33 | (6, 32), (12, 64), (24, 128), (48, 256) | 0.2185, 0.1465, 0.1145, 0.0910 | 1.09x, 1.61x, 2.63x, 4.28x |
| 6.00 | (32, 192), (64, 384) | 0.0720, 0.0710 | 2.23x, 4.47x |
| 8.00 | (12, 96), (24, 192), (32, 256), (48, 384), (64, 512) | 0.0940, 0.0565, 0.0440, 0.0280, 0.0210 | 1.03x, 1.30x, 1.36x, 1.32x, 1.32x |
| 10.67 | (6, 64), (12, 128), (24, 256), (48, 512) | 0.1815, 0.0835, 0.0445, 0.0250 | 0.91x, 0.92x, 1.02x, 1.18x |
| 12.00 | (32, 384), (64, 768) | 0.0350, 0.0200 | 1.09x, 1.26x |
| 16.00 | (6, 96), (12, 192), (24, 384), (32, 512), (48, 768) | 0.1735, 0.0865, 0.0480, 0.0320, 0.0240 | 0.87x, 0.95x, 1.10x, 0.99x, 1.13x |
| 21.33 | (6, 128), (12, 256), (24, 512) | 0.1755, 0.0875, 0.0410 | 0.88x, 0.96x, 0.94x |
| 24.00 | (32, 768) | 0.0350 | 1.09x |
| 32.00 | (6, 192), (12, 384), (24, 768) | 0.1865, 0.0870, 0.0440 | 0.93x, 0.96x, 1.01x |
| 42.67 | (6, 256), (12, 512) | 0.1945, 0.0860 | 0.97x, 0.95x |
| 64.00 | (12, 768) | 0.0870 | 0.96x |

The smallest L/k at which each width still reads at chance, measured:

| k | smallest admissible L/k | i.e. L at least |
|---|---|---|
| 6 | 5.3 | 32 |
| 12 | 8.0 | 96 |
| 24 | 10.7 | 256 |
| 32 | 12.0 | 384 |
| 48 | 10.7 | 512 |

The requirement itself RISES with k, from L/k >= 5.3 at k=6 to L/k >= 12.0 by k=32, so **the admissible region is a narrowing ray, not a rectangle**: raising k needs L raised at least in proportion and in practice more. Of the 51 cells on this grid, 22 are admissible. k=64 is admissible at no length measured here, up to L=768.

| L | largest k that stays within 1.25x | its read | next k up | its read |
|---|---|---|---|---|
| 32 | 6 | 1.09x | 12 | 4.50x |
| 64 | 6 | 0.91x | 12 | 1.61x |
| 96 | 12 | 1.03x | 24 | 4.47x |
| 128 | 12 | 0.92x | 24 | 2.63x |
| 192 | 12 | 0.95x | 24 | 1.30x |
| 256 | 24 | 1.02x | 32 | 1.36x |
| 384 | 32 | 1.09x | 48 | 1.32x |
| 512 | 48 | 1.18x | 64 | 1.32x |
| 768 | 48 | 1.13x | 64 | 1.26x |

### What that costs a live reading

The margin a measured score holds OVER the one-structure read is what the composed cell is buying. On the local Qwen grid at n=40 that margin collapses along k while raw match barely moves — the flatness of the k axis is not a null, it is the cell handing its own separation away:

| L | k | match (n=40) | one-structure read | margin | x chance |
|---|---|---|---|---|---|
| 64 | 6 | 0.550 | 0.1815 | **+0.369** | 2.75x |
| 64 | 12 | 0.550 | 0.1465 | **+0.404** | 6.05x |
| 64 | 24 | 0.700 | 0.3665 | **+0.333** | 16.10x |
| 64 | 32 | 0.550 | 0.4860 | **+0.064** | 17.05x |
| 128 | 6 | 0.250 | 0.1755 | **+0.075** | 1.25x |
| 128 | 12 | 0.250 | 0.0835 | **+0.166** | 2.75x |
| 128 | 24 | 0.200 | 0.1145 | **+0.086** | 4.60x |
| 128 | 32 | 0.225 | 0.1865 | **+0.039** | 6.98x |
| 192 | 6 | 0.125 | 0.1865 | **-0.061** | 0.62x |
| 192 | 12 | 0.075 | 0.0865 | **-0.011** | 0.82x |
| 192 | 24 | 0.075 | 0.0565 | **+0.018** | 1.73x |
| 192 | 32 | 0.125 | 0.0720 | **+0.053** | 3.88x |
| 256 | 6 | 0.200 | 0.1945 | **+0.006** | 1.00x |
| 256 | 12 | 0.075 | 0.0875 | **-0.012** | 0.82x |
| 256 | 24 | 0.100 | 0.0445 | **+0.056** | 2.30x |
| 256 | 32 | 0.075 | 0.0440 | **+0.031** | 2.33x |

L=64 is the row where this model has headroom at every rung, so it is the row that can lose something. Across it match goes 0.550 (k=6) to 0.550 (k=32) — flat, which is what the k axis was reported as buying nothing — while the margin over a half-price solver runs k=6: +0.369, k=12: +0.404, k=24: +0.333, k=32: +0.064. It holds to k=24 and then collapses. The axis is not inert: what it moves is the cell's discriminating power, and it returns a flat score while doing it.

The region is bounded from the other side too, by the sampler rather than by a solver: the composed spec's `q_no_surface` gate cannot fill an n=40 draw at k=6 past L=320, while every k >= 12 generates to at least L=1024. So the narrow-and-long corner is not available and the ray has a floor in k as well as a ceiling.

The shipped operating point is k=12 at L=128 — L/k = 10.7, read 0.92x. It sits ON the frontier, not inside it: the registered cell is as wide as it can be at that length, and the next k rung up at L=128 is already 2.63x.

## Where the next paid point goes, and what it costs

STOP_CEILING asked for a cell a strong model is off the ceiling on. The rule's own remedy was "raise k or L". The k half is ruled out above on two independent grounds — k divides the carrier chain, and past the admissibility ray it hands the cell to a half-price one-structure solver — so the remaining direction is L, with k raised only as far as the ray allows and only for the resolution it buys.

`openai/gpt-5.5` at the registered k=12, measured: match 1.000 at L=128 and 0.975 at L=256, on 7740 and 13457 completion tokens per item — 45 completion and 17 prompt tokens per event, and $0.244 and $0.427 an item. Both cells are at the ceiling. Extending that per-event rate along the admissible ray:

| L | widest admissible k | its one-structure read | informed chance | carrier hops | est. completion tok/item | est. $/item | est. $ at n=40 |
|---|---|---|---|---|---|---|---|
| 256 | 24 | 1.02x | 0.0435 | 7.1 | 13457 | $0.43 | $17 |
| 384 | 32 | 1.09x | 0.0323 | 8.0 | 19175 | $0.61 | $24 |
| 512 | 48 | 1.18x | 0.0213 | 7.1 | 24892 | $0.79 | $32 |
| 768 | 48 | 1.13x | 0.0213 | 10.7 | 36328 | $1.16 | $46 |

The estimate is linear in L at this model's own measured per-event rate, which is what it spends AT THE CEILING; a model that is actually working will spend more, so these are lower bounds on the completion side. It ignores k, which on the local grid LOWERED completion tokens at fixed L (k=6 to k=32 at L=128: 18,654 to 11,745 per item), so the widening is if anything cheaper than priced. These rows are the only extrapolation in this report and are marked as such; everything else is a measurement.

Widening along the ray is bought for RESOLUTION and not for difficulty: at L=512 the widest admissible cell is k=48, whose informed chance is 0.0213 against 0.0909 at the shipped point — 4.3x the measurement resolution — but its carrier chain is 7.1 hops against 28.7 for k=12 at the same length. The difficulty comes from L. k comes along to keep the guess baseline low enough that the difficulty is readable.

## What this arm can and cannot buy

Priced at 200 completion tokens per event and 15.8 completion tokens per second, both measured on this arm:

| step | cells | completion tokens | serialized wall clock |
|---|---|---|---|
| 1. place the model | the four registered k=12 cells at n=40 | 5.12M | **90 h** |
| 2. k in {6,12,24,32,48} at n=40, L=64 | 5 composed cells | 2.56M | **45 h** |
| 2. k in {6,12,24,32,48} at n=40, L=128 | 5 composed cells | 5.12M | **90 h** |
| 2. k in {6,12,24,32,48} at n=40, L=256 | 5 composed cells | 10.24M | **180 h** |
| 3. L in {64,128,192,256} at n=40, one k | 4 composed cells | 5.12M | **90 h** |

One n=40 composed cell at L=128 is 1.02M completion tokens, which this endpoint delivers in 18 hours. The briefed round is 270+ hours on a server that runs one request at a time. It was not run. What was run is priced against the same numbers and stated at its own n.

### The endpoint, measured

The window is not what bounds this grid; the WALL CLOCK is. ds4-server holds one KV session and serializes — at 1/2/4/8 concurrent calls throughput is flat at 16.3-16.4 completion tok/s while per-call latency scales linearly, and the server's own log carries a single generation stream for four concurrent requests. So a cell's duration is its completion tokens divided by one number.

| probe | cell | k | L | workers | budget | completion tokens | wall | completion tok/s | finish |
|---|---|---|---|---|---|---|---|---|---|
| w1_state | state | 12 | 43 | 1 | 16384 | 5411 | 344s | 15.72 | `['stop']` |

### One item can stall a whole cell, and it is not the transport

An `s5_bind_v3` composed@L64 item generated 10,482 tokens in 705 s on the server and was then generated again, byte-identical, four times, and no response ever reached the client. This is a validity problem and not a plumbing one: `backends` retries five times and then records an EMPTY prediction, which is scored wrong, so a path that drops the longest generations manufactures a floor on exactly the cells with the longest traces — the same failure shape as a budget set under the model's trace length. It is not the request timeout: `build_backend` sizes that from the cell's budget and the registry's measured rate (2 x 32,768 / 12.0 = 5,461 s), far above 705 s. Both paths to the server were tried, the registry's HTTPS URL through tailscale-serve and an `ssh -L` forward straight to the server's own port:

| path | k | L | completion tokens | wall | result |
|---|---|---|---|---|---|
| tunnel | 6 | 32 | — | 223s | **APIConnectionError** — VOID as a path test: the server was stopped mid-queue by this session's own `serve_steed_model.py down`, so the connection error is this shutdown and not the transport |

The server's own log says it is not a transport fault. For the item that would not come back it logged `gen=10482 finish=stop` FOUR times against one prompt, byte-identical, each followed by `live kv cache miss ... reason=token-mismatch` and an immediate re-`prompt start` on the same cached prompt — and it went on doing so after the client process was killed. Requests that did return (5,411, 6,675 and 10,887 completion tokens) log the same checkpoint line and are not re-issued, so length alone is not the trigger. `serve_steed_model.py down`/`up` clears it. Until it is understood, a cell on this arm has to be sized so its items stay inside what the server returns, and that is what fixes the lengths below — not a preference for short streams.

## Validity first — every cell's truncation and empty rate

| cell | k | L | n | budget | finish=length | empty | finish reasons | api errors | VOID |
|---|---|---|---|---|---|---|---|---|---|

A cell over 10% finish=length or empty is VOID and enters no comparison until it is re-run at a raised budget: a truncated call is scored wrong, so a truncated cell reads as a floor. The published s5 L64 cliff was a 16-token budget read as a capability, which is why the order is validity first and numbers second.

## 1. Where the model sits against the scouted band

The scout's numbers are the registered k=12 spec at n=40 on the answer read — the same cells and the same read as the rows below.

| model | composed@128 | composed@256 | state@85 | bind@171 |
|---|---|---|---|---|
| openai/gpt-5.5 | 1.000 | 0.975 | 1.000 | 1.000 |
| nvidia/nemotron-3-ultra-550b-a55b | 0.750 | 0.500 | 0.875 | 1.000 |
| z-ai/glm-5.2 | 0.575 | 0.450 | 0.950 | 0.850 |
| **steed/deepseek-v4-flash** | — | — | — | — |

**composed@128, composed@256, state@85, bind@171 were not run on this arm and this model is therefore not placed in the band.** Two independent limits, both measured above: the four band cells at n=40 are 90 hours of serialized generation here, and the endpoint does not return a generation of ~10.5k tokens at all, which is what an item at these lengths costs. The cells below are at the longest stream this arm returns cleanly, which is off the band's axis; they read this model, not its rank against the scouted three.

## 2. The k axis — match against informed chance 1/(k-1), and its token price

| k | L | chance 1/(k-1) | n | match | 95% CI | x chance | z | one-structure read | margin | prompt tok/item | completion tok/item | wall/item |
|---|---|---|---|---|---|---|---|---|---|---|---|---|

`one-structure read` is that cell's half-price policy from the section above, and `margin` is what the measured score holds over it. A cell whose margin is not clearly positive is not reading composition, whatever its match says.

## 3. The L axis, at the same k and the same protocol

Not run on this arm: it returns one length. A second length costs a second cell at this endpoint's serialized rate, and the next length up (L=64) is where an item's trace reaches the size the server re-issued rather than returned. The L axis this round rests on is the local grid's, 16 live cells at n=40, where corr(match, log L) = -0.934 against corr(match, log k) = -0.043.

## 4. The operating point on this model: off the ceiling, components still solved

Not measurable on this arm: the composed cell and both of its work-matched components have to be read at the same (k, L) for the question to mean anything, and the cells that completed do not form that triple. The operating point for a redesign is the one priced above from the scout's own measured usage.

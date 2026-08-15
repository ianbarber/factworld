# The bounded-pad grid, the composed pad's cap, and the pad write's own floor

## What this run establishes

THE COMPOSED CELL'S PAD IS CAPPED, at every registered length on every seed: seed 0 0.525/0.466/0.377; seed 1 0.484/0.427/0.354; seed 2 0.799/0.703/0.558 at L=48/64/96 per token, and seed 0 0.000/0.000/0.000; seed 1 0.000/0.000/0.000; seed 2 0.098/0.057/0.000 PER ITEM — which is the unit the answer is generated in — against components whose pad is perfect on every item at every length including the token-matched ones. So the composed cell's floored answer is a TRACKING result and not a composition one, and the registered rule says so only because the pad gate is applied — without it the same numbers return a composition gap.

THE REGISTERED RESTART DOES NOT DEGRADE THE PAD; it raises it on every seed (composed@48, start to end of `stage4_restart`: seed 0 0.486->0.527; seed 1 0.490->0.501; seed 2 0.657->0.823). The degradation this round was called to investigate belonged to a grouped, answer-ratio-4 readout stage that is not the registered mix.

LENGTH IS NOT THE LEVER. The composed pad is SHORTER than the components' at their token-matched lengths: composed@48 writes 96 pad tokens over an 816-token prompt, against state@80's 160 and bind@132's 264, both written at 1.000. What differs is that every composed event's operand is RESOLVED through the other structure (1.000 of events) and no component event's is (0.000).

THE CAP IS A PER-EVENT RESIDUAL THAT COMPOUNDS, not a missing rule. With the GOLD pad in context the composed per-slot accuracy is flat in length — seed 0: 0.902 0.900 0.897 0.895 0.891 at L=16/32/48/64/96 — while the same slots free-running collapse with it: 0.773 0.626 0.525 0.466 0.377. The median ordinal of each item's FIRST wrong slot is the same at L=48 and L=96 within a seed — seed 0 6/6; seed 1 3/3; seed 2 25/26 — so the pad survives a fixed number of composed events and the stream's length only decides how much of it is wrong by the end.

AND THE RESIDUAL SITS ON THE TWO-HOP WRITE. Teacher-forced, the one pad token that needs the operand resolved and then read through the pointer map (`swap_p0`) scores 0.458-0.964 across seeds and lengths, against the three one-hop tokens of the same events. Split by SOURCE it is 0.398-0.946 where the operand is resolved through the HOLDER map and 0.504-0.982 where it is resolved through the pointer map the write also reads — the first needs both structures, the second is P read twice and is the state component's own carrier depth, and only the first is scored.

THE PAD WRITE HAS A FLOOR AND THE MEASURED PAD IS UNDER IT. The bounded pad hands every policy 2 free live slots, so the registered live-slot rule admits any policy holding 8 of the 12 map cells, and such a policy writes the composed pad at 0.584/0.491/0.390 at L=48/64/96 (3.5/2.9/2.3x the per-slot chance of 1/k = 0.167). The pad width that costs the ANSWER floor nothing costs the PAD floor almost everything: on the answer a partial carry buys 1.05-1.17x chance, on the pad it buys 3-5x, because most pad tokens are one-hop reads of cells the carry holds.

AND THE TWO-HOP TOKEN IS SCORED AGAINST A CLASS THAT MAY NOT COMPOSE IT. The registered live-slot rule is the composed cell's own and is the floor for the three one-hop tokens; the two-hop token takes the COMPONENT cells' conjunct instead — depth <= 1, applied per emitted token — so an admitted row may hold 8 of the 12 map cells and still not perform the resolve-then-read the token is. On the events whose operand is resolved through the OTHER structure, that class reaches 0.3330/0.3472/0.3682 at L=48/64/96 against a per-slot chance of 0.1667, where the class WITHOUT the depth conjunct reaches 0.6032/0.5637/0.5138.

WHICH CONJUNCT IS APPLIED DECIDES THE RESULT, so it is stated rather than assumed. Under the W-only conjunct — the composed cell's own, no depth bound — a row holding 8 of the 12 map cells resolves the operand against a map it holds most of and reaches 0.6032/0.5637/0.5138, and the two seeds that carry a claim are UNDER it at every registered length. The result below rests entirely on the depth conjunct being the right class for this token, and the argument for that is that it is the conjunct both COMPONENT cells' floors are already set by, read on the emission instead of on the policy: a row may not chain two events' contents, and this token is two dependent reads.

THE SCORED READ IS THE TEACHER-FORCED PER-EVENT ONE, and it clears on every seed at every registered length. Per seed at L=48/64/96: seed 0 0.526/0.523/0.513; seed 1 0.486/0.496/0.513; seed 2 0.853/0.842/0.828, against bars 0.4581/0.4696/0.4867. Teacher-forcing removes the COMPOUNDING of a per-event residual over the model's own writes and therefore claims nothing about them: free-running, the same token on the same events reads seed 0 0.279/0.259/0.234; seed 1 0.221/0.217/0.199; seed 2 0.669/0.576/0.454 and clears 6 of 15 cells. The free-running read stays the tracking diagnostic and is printed at every cell.

THE CLAIM IS CONJOINED PER SEED, so it cannot be assembled across models. `components_form_and_read_out` on [0, 1]; `two_hop_clears` on [0, 1, 2] — 2 seed(s) carry all of it ([0, 1]), against 2 required, so a scored composition result on this token EXISTS on this grid. The seed with the best pad is excluded by its own components: its answer is at floor on all eight component cells with a byte-perfect pad, so it reads nothing out and carries no claim about anything downstream of a readout. The result survives its exclusion.

WHAT THAT RESULT IS, exactly: on events whose operand is resolved through the holder map, given the true history, these models write the value that operand points to more often than any policy that may hold 8 of the 12 map cells but may not chain two reads. It is not an ANSWER result and not an end-to-end one — the same grid's answer read returns a tracking gap on the same seeds, and the free-running column above is why.

AND THE HEADROOM IT CLEARS BY IS 0.026 at its tightest — seed 1 at L=64, 0.4960 against a bar of 0.4696 — over the seeds that carry the claim and the registered lengths, and 0.068 at its widest. The floor is a max over an admitted class, so a class that gains a member moves it up: this result stands on that headroom and not on the ratio to chance.

AND THE MARGIN ON THIS READ IS FRACTIONAL, because the additive one is not a bar here. At L=16 the per-slot floor is 0.9271 and the additive rule asks for 1.077 — above 1.0, so no score whatever clears it, at the length where the best measured pad is 0.984. The registered rule is `(a - f)/(1 - f) >= 0.1875`, which is the same 0.15 re-expressed at the answer floor it was calibrated on (0.15 / (1 - 0.2)) and is defined at every floor below 1. Under it composed@16 is buyable at a bar of 0.9408.

## The three cells at every registered length, n=512

`match` / `slot_acc`, `*` clears its bounded-pad floor (z>3.0 and margin>=0.15). Pad width 2, format `moved2`.

| seed | state@17 | state@23 | state@34 | state@80 | bind@31 | bind@41 | bind@62 | bind@132 | composed@48 | composed@64 | composed@96 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 1.000* / 1.000 | 1.000* / 1.000 | 1.000* / 1.000 | 0.998* / 1.000 | 1.000* / 1.000 | 0.998* / 1.000 | 1.000* / 1.000 | 0.994* / 1.000 | 0.197 / 0.525 | 0.141 / 0.466 | 0.191 / 0.377 |
| 1 | 1.000* / 1.000 | 1.000* / 1.000 | 1.000* / 1.000 | 1.000* / 1.000 | 1.000* / 1.000 | 1.000* / 1.000 | 1.000* / 1.000 | 1.000* / 1.000 | 0.174 / 0.484 | 0.195 / 0.427 | 0.193 / 0.354 |
| 2 | 0.170 / 1.000 | 0.203 / 1.000 | 0.191 / 1.000 | 0.191 / 1.000 | 0.152 / 1.000 | 0.152 / 1.000 | 0.176 / 1.000 | 0.150 / 1.000 | 0.174 / 0.799 | 0.201 / 0.703 | 0.213 / 0.558 |
| floor | 0.2000 | 0.2090 | 0.2000 | 0.2090 | 0.2000 | 0.2000 | 0.2000 | 0.2000 | 0.2285 | 0.2112 | 0.2168 |

## The same cells BEFORE the restart, n=128

| seed | state@17 | state@23 | state@34 | state@80 | bind@31 | bind@41 | bind@62 | bind@132 | composed@48 | composed@64 | composed@96 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 0.984* / 1.000 | 0.977* / 1.000 | 0.969* / 1.000 | 0.969* / 1.000 | 0.977* / 1.000 | 0.922* / 1.000 | 0.914* / 1.000 | 0.914* / 1.000 | 0.211 / 0.486 | 0.219 / 0.428 | 0.219 / 0.361 |
| 1 | 1.000* / 1.000 | 1.000* / 1.000 | 1.000* / 1.000 | 1.000* / 1.000 | 1.000* / 1.000 | 1.000* / 1.000 | 1.000* / 1.000 | 1.000* / 1.000 | 0.148 / 0.490 | 0.242 / 0.426 | 0.172 / 0.354 |
| 2 | 0.133 / 1.000 | 0.148 / 1.000 | 0.203 / 1.000 | 0.133 / 0.997 | 0.172 / 1.000 | 0.133 / 1.000 | 0.125 / 1.000 | 0.164 / 1.000 | 0.203 / 0.657 | 0.188 / 0.578 | 0.156 / 0.461 |

## The registered rule, applied to these numbers

- components FORM at their own registered lengths: {'state': {17: 2, 23: 2, 34: 2}, 'bind': {31: 2, 41: 2, 62: 2}, 'composed': {48: 0, 64: 0, 96: 0}} (needs 2 seeds at every length)
- matched-cost control at composed@48: {'state': True, 'bind': True}
- composed cell clears: {48: 0, 64: 0, 96: 0}
- composed pad is perfect on 0.99 of ITEMS on: {48: 0, 64: 0, 96: 0} seeds per length -> pad_tracked=False (per seed at L=48/64/96: seed 0 0.000/0.000/0.000; seed 1 0.000/0.000/0.000; seed 2 0.098/0.057/0.000)
- composed GOLD-PAD answer: not measured in this run
- PER-SEED conjunction of the gates {'components_form': [0, 1], 'pad_tracks': []} -> 0 seed(s) carry the whole claim ({0: False, 1: False, 2: False}); needs 2

**Without the pad gate the rule returns `V1_COMPOSITION_GAP`. With it, `V6_TRACKING_GAP`.**

both components form and the composed cell is at floor, but the composed cell does not write its own pad at the level the components reach: items perfect >= 0.99 on {48: 0, 64: 0, 96: 0} seeds per length, against components whose pad is perfect on every item everywhere. This read scores the answer the model gives FROM ITS OWN PAD, so a floored answer here is equally consistent with the composition being hard and with the model being unable to hold the state the pad gave it room for, and only the second is measured. No composition claim is available until the composed pad reaches component level with the answer still at floor.

## The PAD WRITE against its own floors

Two floors, because the pad has two kinds of token in it. The REGISTERED CLASS is the composed cell's own: live slots `W - pad <= max(k, m) + 1`, steps no more than the algorithm pays to produce the pad — at pad 2 any policy holding 8 of the 12 map cells, and not the algorithm itself (12 + scratch). It is the floor for the three ONE-HOP tokens. The TWO-HOP token takes the COMPONENT cells' conjunct instead, `depth <= 1` applied per emitted token, so an admitted row may hold most of both maps and still not perform the resolve-then-read that this token is. Chance for a per-slot read is `1/k` — every pad token is an agent name — and not the answer read's `1/(k-1)`.

`swap_p0` is split by SOURCE and only the cross half is scored: on a same-source swap both reads are of P, which a row holding P alone performs exactly and which is the state component's own carrier depth. Pooling the two lets a model that does the same-source write and floors on the cross one read as though it composed.

| cell | chance | per-slot floor (registered class) | row | swap_p0\|cross, one-hop floor | row | same-source, one-hop | swap_p0\|cross unrestricted |
|---|---|---|---|---|---|---|---|
| composed@16 | 0.1667 | 0.9366 (5.62x) | `pad_carry_P4B4_first[scored]` | 0.2382 (1.43x) | `pad_carry_P0B0_first:copy_prev[scored]` | 0.2473 | 0.8903 |
| composed@32 | 0.1667 | 0.8130 (4.88x) | `pad_carry_P4B4_first[scored]` | 0.3061 (1.84x) | `pad_carry_P0B0_first:gold_pad_fixed@swap:cross[-1]p1[scored]` | 0.2918 | 0.6994 |
| composed@48 | 0.1667 | 0.7363 (4.42x) | `pad_carry_P3B5_first[scored]` | 0.3330 (2.00x) | `pad_carry_P0B0_first:gold_pad_fixed@swap:cross[-1]p1[disjoint]` | 0.3037 | 0.6032 |
| composed@64 | 0.1667 | 0.7031 (4.22x) | `pad_carry_P2B6_recent[scored]` | 0.3472 (2.08x) | `pad_carry_P0B0_first:gold_pad_fixed@swap:cross[-1]p1[scored]` | 0.3202 | 0.5637 |
| composed@96 | 0.1667 | 0.6853 (4.11x) | `pad_carry_P2B6_recent[scored]` | 0.3682 (2.21x) | `pad_carry_P0B0_first:gold_pad_fixed@swap:cross[-1]p1[disjoint]` | 0.3167 | 0.5138 |

Floors are on the teacher_forced read, which is the scored one; the same table on the free-running read is in the JSON. A floor is measured on the exact scored items AND on a disjoint pool with the larger operative, since a max over 78 rows carries an upward selection bias at small n.

### The gold-pad emission family, closed by a rule

Under the teacher-forced read the context IS the gold pad, so copying one token out of it at an address that does not have to be searched for costs 0 hops, 0 live slots and 0 steps. The admitted class therefore contains EVERY fixed positional read of the pad, and the floor maximises over the whole family rather than over registered names (`validity.s5_bind_v3_pad_gold_reads`). An address is a block CLASS the row's emission is already partitioned by — the event's kind and its source, which it reads off the event line anyway — an INDEX that is a constant in that class's prefix under either canonical indexing (the d-th most recent, the a-th from the start), and a token POSITION. The sweep is exhaustive at each length: a stream of L events has L blocks, so the family is finite and the max is attained inside what is swept by construction.

A per-event BACKWARD SCAN is not a fixed positional read and stays excluded. Its block is found by MATCHING the current event's operand against earlier events, one comparison per event passed, so the address is not a constant and the cost is per event; the step conjunct excludes it and it is priced under the scored read below rather than argued away.

| cell | addresses swept | best address | its score | leg | back-offset 1 / 2 / 3 / 4 | max at d>=9 |
|---|---|---|---|---|---|---|
| composed@16 | 398 | `swap[-1]p1` | 0.2305 | scored | 0.2305 / 0.1401 / 0.0910 / 0.0555 | 0.0013 |
| composed@32 | 739 | `swap:cross[-1]p1` | 0.3061 | scored | 0.3061 / 0.1344 / 0.0893 / 0.0537 | 0.0021 |
| composed@48 | 1091 | `swap:cross[-1]p1` | 0.3330 | disjoint | 0.3330 / 0.1498 / 0.1153 / 0.0908 | 0.0156 |
| composed@64 | 1346 | `swap:cross[-1]p1` | 0.3472 | scored | 0.3472 / 0.1616 / 0.1347 / 0.1098 | 0.0412 |
| composed@96 | 1939 | `swap:cross[-1]p1` | 0.3682 | disjoint | 0.3682 / 0.1661 / 0.1441 / 0.1301 | 0.0772 |

The max sits at the most recent block of its class at every length and the profile decays away from it: what the family buys is ADJACENCY, not depth into the pad. At composed@96 the coarsest class reads at most 0.1604 at offsets 9 and beyond, against a per-slot chance of 0.1667 — far into the pad a fixed read is an uninformative draw from its marginal, which is `uniform` and already in the class. Both legs are swept and the larger is operative, so the address that wins on the scored items does not have to be the one that wins on the disjoint pool for the floor to cover it.

### The excluded backward scan, priced on the read the model is scored on

`pad_scan_last_write` carries P in full and recovers a cross swap's operand by scanning back to the last give that wrote the referenced object. Its per-event cost does not grow with L, so an L-independence rule would admit it; it is excluded on TOTAL STEPS (~2m/p_give per swap against the algorithm's 6) and its score is printed so the exclusion is a judgement about cost and not about the number.

The model is scored teacher-forced on the CROSS partition, so that is the reading printed first; the free-running pooled figure is the different quantity it is and is kept beside it rather than standing in for it.

| cell | admitted | scored read (teacher-forced) swap_p0\|cross | teacher-forced swap_p0 pooled | free-running swap_p0 pooled |
|---|---|---|---|---|
| composed@16 | False | 0.8244 | 0.8907 | 0.8799 |
| composed@32 | False | 0.5930 | 0.7116 | 0.6730 |
| composed@48 | False | 0.4651 | 0.5849 | 0.5131 |
| composed@64 | False | 0.4184 | 0.5353 | 0.4532 |
| composed@96 | False | 0.3370 | 0.4501 | 0.3533 |

AGAINST THE SEEDS THAT CARRY THE CLAIM it reads above every one of them at L=16/32 and below every one of them at L=48/64/96. The exclusion holds at all of them either way: it rests on the step conjunct and not on where the number lands, which is why the number is printed.

### The scored read, per seed and per length

Teacher-forced per event, n=512, under `clears_headroom` (z>3.0 and `(a-f)/(1-f) >= 0.1875`). Teacher-forcing removes the COMPOUNDING of a per-event residual over the model's own writes and therefore claims nothing about them; the free-running column beside it is the tracking diagnostic.

| seed | cell | swap_p0\|cross | floor | bar | clears | same-source | free-running cross | its floor | clears |
|---|---|---|---|---|---|---|---|---|---|
| 0 | composed@16 | 0.5294 | 0.2382 | 0.3811 | yes | 0.6040 | 0.4351 | 0.2382 | yes |
| 0 | composed@32 | 0.5311 | 0.3061 | 0.4362 | yes | 0.5999 | 0.3249 | 0.1861 | no |
| 0 | composed@48 | 0.5256 | 0.3330 | 0.4581 | yes | 0.5979 | 0.2789 | 0.1864 | no |
| 0 | composed@64 | 0.5233 | 0.3472 | 0.4696 | yes | 0.5982 | 0.2586 | 0.1818 | no |
| 0 | composed@96 | 0.5131 | 0.3682 | 0.4867 | yes | 0.5784 | 0.2336 | 0.1740 | no |
| 1 | composed@16 | 0.3977 | 0.2382 | 0.3811 | yes | 0.5161 | 0.2744 | 0.2382 | no |
| 1 | composed@32 | 0.4720 | 0.3061 | 0.4362 | yes | 0.5041 | 0.2244 | 0.1861 | no |
| 1 | composed@48 | 0.4856 | 0.3330 | 0.4581 | yes | 0.5172 | 0.2206 | 0.1864 | no |
| 1 | composed@64 | 0.4960 | 0.3472 | 0.4696 | yes | 0.5160 | 0.2175 | 0.1818 | no |
| 1 | composed@96 | 0.5135 | 0.3682 | 0.4867 | yes | 0.5113 | 0.1993 | 0.1740 | no |
| 2 | composed@16 | 0.9464 | 0.2382 | 0.3811 | yes | 0.9817 | 0.9342 | 0.2382 | yes |
| 2 | composed@32 | 0.8865 | 0.3061 | 0.4362 | yes | 0.9628 | 0.7863 | 0.1861 | yes |
| 2 | composed@48 | 0.8534 | 0.3330 | 0.4581 | yes | 0.9372 | 0.6689 | 0.1864 | yes |
| 2 | composed@64 | 0.8420 | 0.3472 | 0.4696 | yes | 0.9222 | 0.5764 | 0.1818 | yes |
| 2 | composed@96 | 0.8282 | 0.3682 | 0.4867 | yes | 0.9185 | 0.4541 | 0.1740 | yes |

### The claim, conjoined per seed

A seed counts only where EVERY gate holds for THAT seed. Counted apart, three gates are satisfied by a run in which no single model satisfies two of them.

| gate | seeds |
|---|---|
| `components_form_and_read_out` | [0, 1] |
| `two_hop_clears` | [0, 1, 2] |
| **all of them** | **[0, 1]** |

2 seed(s) carry the whole claim at the registered lengths [48, 64, 96]; 2 are required, so the claim HOLDS.

### The saturation control

A saturation control asks whether the measurement can register a clear at all. NO MODEL CONTROL EXISTS ON THIS READ and the reason is structural, not a gap in the run: a component cell renders every operand by NAME, so the scored token has ZERO events there (and its pad-write floor is 1.0000 at bind@62, 1.0000 at state@34). The quantity is defined on the composed cell and nowhere else, so no cell carries a model that is known good on other evidence AND has the token to write.

What can be built is a POLICY control at the other end of the same scale. The composed cell's own algorithm is a row in this family — `pad_carry_P6B6_first`, carrying both maps — and it is excluded by the live-slot conjunct at W = 13 against the bound 9. On the exact scored items it writes the two-hop token at 1.0000 on n = 1549 events, against an admitted class that reaches 0.2382 at composed@16. That exercises the SCORER over the whole range; it does not exercise the decode, and the difference is what the missing model control would have bought.

WHAT ITS ABSENCE COSTS, stated rather than absorbed: this read is uncalibrated against FALSE NEGATIVES. A floored two-hop score on some future model could be the measurement rather than the model, and nothing here would separate them. It does not weaken a clear — a clear is the model emitting the token more often than any admitted policy does, on items the policies were measured on — and the run reported here has no null on this token to protect.

## Pad accuracy ACROSS the stage

**seed 0, `stage3_composition`** (every 1750 steps, n=128)

| cell | 0 | 1750 | 3500 | 5250 | 7000 | 8750 |
|---|---|---|---|---|---|---|
| state@34 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| bind@62 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| composed@48 | 0.473 | 0.477 | 0.483 | 0.483 | 0.492 | 0.486 |
| composed@64 | 0.428 | 0.434 | 0.432 | 0.437 | 0.434 | 0.428 |
| composed@96 | 0.356 | 0.347 | 0.351 | 0.355 | 0.363 | 0.361 |

**seed 0, `stage4_restart`** (every 500 steps, n=128)

| cell | 0 | 500 | 1000 | 1500 | 2000 | 2500 | 3000 |
|---|---|---|---|---|---|---|---|
| state@34 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| bind@62 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| composed@48 | 0.486 | 0.489 | 0.499 | 0.513 | 0.505 | 0.514 | 0.527 |
| composed@64 | 0.428 | 0.422 | 0.435 | 0.441 | 0.444 | 0.464 | 0.461 |
| composed@96 | 0.361 | 0.357 | 0.351 | 0.361 | 0.368 | 0.369 | 0.366 |

**seed 1, `stage3_composition`** (every 1750 steps, n=128)

| cell | 0 | 1750 | 3500 | 5250 | 7000 | 8750 |
|---|---|---|---|---|---|---|
| state@34 | 1.000 | 0.999 | 1.000 | 1.000 | 1.000 | 1.000 |
| bind@62 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| composed@48 | 0.469 | 0.448 | 0.474 | 0.471 | 0.478 | 0.490 |
| composed@64 | 0.428 | 0.403 | 0.428 | 0.423 | 0.424 | 0.426 |
| composed@96 | 0.353 | 0.332 | 0.351 | 0.350 | 0.348 | 0.354 |

**seed 1, `stage4_restart`** (every 500 steps, n=128)

| cell | 0 | 500 | 1000 | 1500 | 2000 | 2500 | 3000 |
|---|---|---|---|---|---|---|---|
| state@34 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| bind@62 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| composed@48 | 0.490 | 0.480 | 0.494 | 0.499 | 0.495 | 0.505 | 0.501 |
| composed@64 | 0.426 | 0.417 | 0.424 | 0.422 | 0.446 | 0.433 | 0.435 |
| composed@96 | 0.354 | 0.343 | 0.345 | 0.349 | 0.356 | 0.349 | 0.353 |

**seed 2, `stage3_composition`** (every 1750 steps, n=128)

| cell | 0 | 1750 | 3500 | 5250 | 7000 | 8750 |
|---|---|---|---|---|---|---|
| state@34 | 1.000 | 1.000 | 0.999 | 1.000 | 1.000 | 1.000 |
| bind@62 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| composed@48 | 0.438 | 0.431 | 0.554 | 0.623 | 0.660 | 0.657 |
| composed@64 | 0.383 | 0.363 | 0.470 | 0.536 | 0.552 | 0.578 |
| composed@96 | 0.317 | 0.305 | 0.376 | 0.411 | 0.439 | 0.461 |

**seed 2, `stage4_restart`** (every 500 steps, n=128)

| cell | 0 | 500 | 1000 | 1500 | 2000 | 2500 | 3000 |
|---|---|---|---|---|---|---|---|
| state@34 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| bind@62 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| composed@48 | 0.657 | 0.648 | 0.707 | 0.754 | 0.795 | 0.807 | 0.823 |
| composed@64 | 0.578 | 0.577 | 0.614 | 0.652 | 0.683 | 0.697 | 0.704 |
| composed@96 | 0.461 | 0.448 | 0.504 | 0.532 | 0.546 | 0.558 | 0.572 |

## Is the composed pad simply longer? (no model)

| cell | events | pad tokens | prompt tokens | pad/prompt | resolved operand |
|---|---|---|---|---|---|
| state@17 | 17 | 34 | 221 | 0.154 | 0.000 |
| state@23 | 23 | 46 | 281 | 0.164 | 0.000 |
| state@34 | 34 | 68 | 391 | 0.174 | 0.000 |
| state@80 | 80 | 160 | 851 | 0.188 | 0.000 |
| state@108 | 108 | 216 | 1131 | 0.191 | 0.000 |
| bind@31 | 31 | 62 | 268 | 0.231 | 0.000 |
| bind@41 | 41 | 82 | 338 | 0.243 | 0.000 |
| bind@62 | 62 | 124 | 485 | 0.256 | 0.000 |
| bind@132 | 132 | 264 | 975 | 0.271 | 0.000 |
| composed@48 | 48 | 96 | 816 | 0.118 | 1.000 |
| composed@64 | 64 | 128 | 1058 | 0.121 | 1.000 |
| composed@96 | 96 | 192 | 1538 | 0.125 | 1.000 |

## Is the per-event update missing, or only the closed loop?

Slot accuracy with the GOLD pad in context, one forward per item. A diagnostic: teacher-forced accuracy is not a score on the task, because the true history is what the task withholds.

On the COMPOSED rows `swap_p0` is the only two-hop token — the operand is resolved through the holder map and then read through the pointer map — and the other three are one hop. On a COMPONENT row every operand is NAMED, so its `swap_p0` is a one-hop read and is not the same quantity.

| seed | cell | per_slot | give_p0 | give_p1 | swap_p0 | swap_p1 |
|---|---|---|---|---|---|---|
| 0 | state@34 | 1.000 | — | — | 1.000 | 1.000 |
| 0 | bind@62 | 1.000 | 1.000 | 1.000 | — | — |
| 0 | composed@16 | 0.902 | 0.984 | 1.000 | 0.567 | 0.947 |
| 0 | composed@32 | 0.900 | 0.969 | 1.000 | 0.566 | 0.933 |
| 0 | composed@48 | 0.897 | 0.961 | 1.000 | 0.562 | 0.929 |
| 0 | composed@64 | 0.895 | 0.961 | 0.999 | 0.561 | 0.924 |
| 0 | composed@96 | 0.891 | 0.958 | 0.999 | 0.546 | 0.918 |
| 1 | state@34 | 1.000 | — | — | 1.000 | 1.000 |
| 1 | bind@62 | 1.000 | 1.000 | 1.000 | — | — |
| 1 | composed@16 | 0.887 | 0.990 | 1.000 | 0.458 | 0.965 |
| 1 | composed@32 | 0.895 | 0.983 | 1.000 | 0.488 | 0.955 |
| 1 | composed@48 | 0.896 | 0.977 | 1.000 | 0.502 | 0.954 |
| 1 | composed@64 | 0.895 | 0.976 | 0.999 | 0.506 | 0.949 |
| 1 | composed@96 | 0.896 | 0.975 | 0.999 | 0.512 | 0.949 |
| 2 | state@34 | 1.000 | — | — | 1.000 | 1.000 |
| 2 | bind@62 | 1.000 | 1.000 | 1.000 | — | — |
| 2 | composed@16 | 0.992 | 0.998 | 0.999 | 0.964 | 0.998 |
| 2 | composed@32 | 0.982 | 0.993 | 0.997 | 0.925 | 0.993 |
| 2 | composed@48 | 0.973 | 0.987 | 0.994 | 0.896 | 0.987 |
| 2 | composed@64 | 0.969 | 0.985 | 0.992 | 0.882 | 0.987 |
| 2 | composed@96 | 0.965 | 0.981 | 0.989 | 0.873 | 0.983 |

## The same slots free-running, decomposed

The grid's own read, instrumented per slot instead of pooled. `swap_p0` is the only two-hop token: the operand is resolved through the holder map and then read through the pointer map.

On the COMPOSED rows `swap_p0` is the only two-hop token — the operand is resolved through the holder map and then read through the pointer map — and the other three are one hop. On a COMPONENT row every operand is NAMED, so its `swap_p0` is a one-hop read and is not the same quantity.

| seed | cell | per_slot | give_p0 | give_p1 | swap_p0 | swap_p1 | items perfect | median first error |
|---|---|---|---|---|---|---|---|---|
| 0 | state@34 | 1.000 | — | — | 1.000 | 1.000 | 1.000 | None |
| 0 | bind@62 | 1.000 | 1.000 | 1.000 | — | — | 1.000 | None |
| 0 | composed@16 | 0.773 | 0.870 | 0.955 | 0.453 | 0.642 | 0.068 | 6 |
| 0 | composed@32 | 0.626 | 0.695 | 0.807 | 0.345 | 0.462 | 0.002 | 6 |
| 0 | composed@48 | 0.525 | 0.574 | 0.680 | 0.300 | 0.380 | 0.000 | 6 |
| 0 | composed@64 | 0.466 | 0.508 | 0.594 | 0.279 | 0.342 | 0.000 | 6 |
| 0 | composed@96 | 0.377 | 0.403 | 0.470 | 0.250 | 0.290 | 0.000 | 6 |
| 1 | state@34 | 1.000 | — | — | 1.000 | 1.000 | 1.000 | None |
| 1 | bind@62 | 1.000 | 1.000 | 1.000 | — | — | 1.000 | None |
| 1 | composed@16 | 0.704 | 0.821 | 0.932 | 0.309 | 0.543 | 0.023 | 3 |
| 1 | composed@32 | 0.584 | 0.664 | 0.785 | 0.263 | 0.403 | 0.000 | 3 |
| 1 | composed@48 | 0.484 | 0.534 | 0.644 | 0.244 | 0.342 | 0.000 | 3 |
| 1 | composed@64 | 0.427 | 0.468 | 0.560 | 0.239 | 0.298 | 0.000 | 3 |
| 1 | composed@96 | 0.354 | 0.381 | 0.446 | 0.227 | 0.266 | 0.000 | 3 |
| 2 | state@34 | 1.000 | — | — | 1.000 | 1.000 | 1.000 | None |
| 2 | bind@62 | 1.000 | 1.000 | 1.000 | — | — | 1.000 | None |
| 2 | composed@16 | 0.984 | 0.991 | 0.997 | 0.951 | 0.982 | 0.799 | 13 |
| 2 | composed@32 | 0.912 | 0.935 | 0.959 | 0.818 | 0.881 | 0.361 | 21 |
| 2 | composed@48 | 0.799 | 0.818 | 0.860 | 0.693 | 0.760 | 0.098 | 25 |
| 2 | composed@64 | 0.703 | 0.721 | 0.771 | 0.596 | 0.653 | 0.057 | 24 |
| 2 | composed@96 | 0.558 | 0.573 | 0.618 | 0.470 | 0.508 | 0.000 | 26 |

## Where in the stream the pad breaks

Free-running per-slot accuracy at event ordinal, as a fraction of the stream.

| seed | cell | 0% | 12% | 25% | 50% | 75% | 100% |
|---|---|---|---|---|---|---|---|
| 0 | composed@16 | 0.962 | 0.938 | 0.894 | 0.778 | 0.704 | 0.476 |
| 0 | composed@32 | 0.977 | 0.898 | 0.802 | 0.591 | 0.461 | 0.317 |
| 0 | composed@48 | 0.972 | 0.847 | 0.696 | 0.470 | 0.361 | 0.255 |
| 0 | composed@64 | 0.959 | 0.787 | 0.618 | 0.393 | 0.297 | 0.243 |
| 0 | composed@96 | 0.970 | 0.740 | 0.477 | 0.284 | 0.229 | 0.187 |
| 1 | composed@16 | 0.887 | 0.863 | 0.813 | 0.722 | 0.637 | 0.405 |
| 1 | composed@32 | 0.923 | 0.818 | 0.763 | 0.544 | 0.451 | 0.314 |
| 1 | composed@48 | 0.901 | 0.781 | 0.626 | 0.428 | 0.349 | 0.236 |
| 1 | composed@64 | 0.902 | 0.698 | 0.567 | 0.358 | 0.268 | 0.222 |
| 1 | composed@96 | 0.909 | 0.644 | 0.437 | 0.260 | 0.219 | 0.183 |
| 2 | composed@16 | 1.000 | 1.000 | 0.999 | 0.996 | 0.980 | 0.921 |
| 2 | composed@32 | 1.000 | 1.000 | 0.993 | 0.949 | 0.845 | 0.679 |
| 2 | composed@48 | 1.000 | 0.997 | 0.979 | 0.845 | 0.648 | 0.472 |
| 2 | composed@64 | 1.000 | 0.997 | 0.943 | 0.719 | 0.484 | 0.332 |
| 2 | composed@96 | 1.000 | 0.975 | 0.843 | 0.473 | 0.282 | 0.216 |

## Is the pad being traded away by answer supervision?

Equal continuations of 600 steps at lr 0.0003 under 3 mixes (composed_only, no_answer_docs, registered), composed `slot_acc` before -> after.

| seed | mix | composed@48 |
|---|---|---|
| 0 | `registered` | 0.525 -> 0.550 |
| 0 | `no_answer_docs` | 0.525 -> 0.556 |
| 0 | `composed_only` | 0.525 -> 0.568 |
| 1 | `registered` | 0.484 -> 0.485 |
| 1 | `no_answer_docs` | 0.484 -> 0.490 |
| 1 | `composed_only` | 0.484 -> 0.483 |
| 2 | `registered` | 0.799 -> 0.790 |
| 2 | `no_answer_docs` | 0.799 -> 0.793 |
| 2 | `composed_only` | 0.799 -> 0.795 |

## Does more composed training close it?

3000 steps at lr 0.0003 on the composed cell ALONE, from the registered grid's own final checkpoint. This is the restart's whole budget spent on nothing but the cell whose pad is capped.

| seed | mix | composed@48 | composed@96 |
|---|---|---|---|
| 0 | `composed_only` | 0.525 -> 0.690 | 0.377 -> 0.484 |
| 1 | `composed_only` | 0.484 -> 0.509 | 0.354 -> 0.360 |
| 2 | `composed_only` | 0.799 -> 0.832 | 0.558 -> 0.602 |


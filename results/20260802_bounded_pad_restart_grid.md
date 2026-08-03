# The bounded-pad grid, the composed pad's cap, and the pad write's own floor

## What this run establishes

THE COMPOSED CELL'S PAD IS CAPPED, at every registered length on every seed: seed 0 0.525/0.466/0.377; seed 1 0.484/0.427/0.354; seed 2 0.799/0.703/0.558 at L=48/64/96 per token, and seed 0 0.000/0.000/0.000; seed 1 0.000/0.000/0.000; seed 2 0.098/0.057/0.000 PER ITEM — which is the unit the answer is generated in — against components whose pad is perfect on every item at every length including the token-matched ones. So the composed cell's floored answer is a TRACKING result and not a composition one, and the registered rule says so only because the pad gate is applied — without it the same numbers return a composition gap.

THE REGISTERED RESTART DOES NOT DEGRADE THE PAD; it raises it on every seed (composed@48, start to end of `stage4_restart`: seed 0 0.486->0.527; seed 1 0.490->0.501; seed 2 0.657->0.823). The degradation this round was called to investigate belonged to a grouped, answer-ratio-4 readout stage that is not the registered mix.

LENGTH IS NOT THE LEVER. The composed pad is SHORTER than the components' at their token-matched lengths: composed@48 writes 96 pad tokens over an 816-token prompt, against state@80's 160 and bind@132's 264, both written at 1.000. What differs is that every composed event's operand is RESOLVED through the other structure (1.000 of events) and no component event's is (0.000).

THE CAP IS A PER-EVENT RESIDUAL THAT COMPOUNDS, not a missing rule. With the GOLD pad in context the composed per-slot accuracy is flat in length — seed 0: 0.902 0.900 0.897 0.895 0.891 at L=16/32/48/64/96 — while the same slots free-running collapse with it: 0.773 0.626 0.525 0.466 0.377. The median ordinal of each item's FIRST wrong slot is the same at L=48 and L=96 within a seed — seed 0 6/6; seed 1 3/3; seed 2 25/26 — so the pad survives a fixed number of composed events and the stream's length only decides how much of it is wrong by the end.

AND THE RESIDUAL SITS ON THE TWO-HOP WRITE. Teacher-forced, the one pad token that needs the operand resolved through the holder map and then read through the pointer map (`swap_p0`) scores 0.458-0.964 across seeds and lengths, against the three one-hop tokens of the same events. That is where the composition is, and the next section is what it is worth against a floor.

THE PAD WRITE HAS A FLOOR AND THE MEASURED PAD IS UNDER IT. The bounded pad hands every policy 2 free live slots, so the registered live-slot rule admits any policy holding 8 of the 12 map cells, and such a policy writes the composed pad at 0.584/0.491/0.390 at L=48/64/96 (3.5/2.9/2.3x the per-slot chance of 1/k = 0.167). The pad width that costs the ANSWER floor nothing costs the PAD floor almost everything: on the answer a partial carry buys 1.05-1.17x chance, on the pad it buys 3-5x, because most pad tokens are one-hop reads of cells the carry holds.

SO THE TWO-HOP TOKEN DOES NOT REGISTER A COMPOSITION EITHER. Free-running — the read the grid scores — `swap_p0`'s floor is 0.536/0.454/0.363, and the same one-structure carry sets it: a policy that never composes the two hops correctly still gets that token right 3-5x chance by resolving the operand against a map it holds only part of. The one-hop SUB-class reaches only 0.177-0.189 there, so the token is two-hop work — but the class that decides a floor is the registered one, and it reaches the model.

AND IT DOES NOT FORM. Of 3 seeds, seed 2 clears at L=32/48/64/96, against the 2 seeds FORMS requires at EVERY registered length. The seed(s) that clear — [2] — are the ones whose ANSWER is at floor on every cell including both components, so the pad write separates on exactly the seed that reads nothing out.

AND THE SHORT-STREAM ESCAPE IS CLOSED ON THIS AXIS TOO. At L=16, where the best seed writes 0.984 of the pad, the floor is 0.9271 and the CLEARS bar is 1.077 — above 1.0, so no score at that length can clear it. The pad-write read is unbuyable at exactly the lengths where the pad is written.

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

**Without the pad gate the rule returns `V1_COMPOSITION_GAP`. With it, `V6_TRACKING_GAP`.**

both components form and the composed cell is at floor, but the composed cell does not write its own pad at the level the components reach: items perfect >= 0.99 on {48: 0, 64: 0, 96: 0} seeds per length, against components whose pad is perfect on every item everywhere. This read scores the answer the model gives FROM ITS OWN PAD, so a floored answer here is equally consistent with the composition being hard and with the model being unable to hold the state the pad gave it room for, and only the second is measured. No composition claim is available until the composed pad reaches component level with the answer still at floor.

## The PAD WRITE against its own floor

The class is the registered one, scored on the pad instead of on the answer: live slots `W - pad <= max(k, m) + 1`, steps no more than the cell's own algorithm pays to produce the pad. At pad 2 that admits any policy holding 8 of the 12 map cells and excludes the cell's own algorithm (12 + scratch). Chance for a per-slot read is `1/k` — every pad token is an agent name — and not the answer read's `1/(k-1)`.

| cell | chance | per-slot floor | row | swap_p0 floor | swap_p0, one-hop sub-class | CLEARS bar |
|---|---|---|---|---|---|---|
| composed@16 | 0.1667 | 0.9271 (5.56x) | `pad_carry_P4B4_first[scored]` | 0.9096 | 0.2291 | 1.077 — above 1.0, unbuyable |
| composed@32 | 0.1667 | 0.7390 (4.43x) | `pad_carry_P4B4_first[scored]` | 0.6793 | 0.1891 | 0.889 |
| composed@48 | 0.1667 | 0.5835 (3.50x) | `pad_carry_P4B4_first[scored]` | 0.5363 | 0.1888 | 0.734 |
| composed@64 | 0.1667 | 0.4906 (2.94x) | `pad_carry_P4B4_first[scored]` | 0.4541 | 0.1816 | 0.641 |
| composed@96 | 0.1667 | 0.3898 (2.34x) | `pad_carry_P4B4_first[disjoint]` | 0.3626 | 0.1767 | 0.540 |

Every measured seed against it, n=512, under the registered `clears` (z>3.0 and margin>=0.15).

| seed | cell | per-slot | floor | clears | swap_p0 | floor | clears |
|---|---|---|---|---|---|---|---|
| 0 | composed@16 | 0.773 | 0.9271 | no | 0.453 | 0.9096 | no |
| 0 | composed@32 | 0.626 | 0.7390 | no | 0.345 | 0.6793 | no |
| 0 | composed@48 | 0.525 | 0.5835 | no | 0.300 | 0.5363 | no |
| 0 | composed@64 | 0.466 | 0.4906 | no | 0.279 | 0.4541 | no |
| 0 | composed@96 | 0.377 | 0.3898 | no | 0.250 | 0.3626 | no |
| 1 | composed@16 | 0.704 | 0.9271 | no | 0.309 | 0.9096 | no |
| 1 | composed@32 | 0.584 | 0.7390 | no | 0.263 | 0.6793 | no |
| 1 | composed@48 | 0.484 | 0.5835 | no | 0.244 | 0.5363 | no |
| 1 | composed@64 | 0.427 | 0.4906 | no | 0.239 | 0.4541 | no |
| 1 | composed@96 | 0.354 | 0.3898 | no | 0.227 | 0.3626 | no |
| 2 | composed@16 | 0.984 | 0.9271 | no | 0.951 | 0.9096 | no |
| 2 | composed@32 | 0.912 | 0.7390 | yes | 0.818 | 0.6793 | no |
| 2 | composed@48 | 0.799 | 0.5835 | yes | 0.693 | 0.5363 | yes |
| 2 | composed@64 | 0.703 | 0.4906 | yes | 0.596 | 0.4541 | no |
| 2 | composed@96 | 0.558 | 0.3898 | yes | 0.470 | 0.3626 | no |

AND THE PAD-WRITE READ SUPPORTS NO POSITIVE CONTROL. A component cell has ONE structure, so the one-structure bound admits the component's own pad algorithm and its floor is 1.0000 at bind@62, 1.0000 at state@34 — the same argument that leaves the composed cell unfloorable under a `k + m` wide pad, one cell over. A component's perfect pad cannot clear anything, so this read has no cell on which a working model demonstrates that the measurement works.

AND THE TEACHER-FORCED READ HAS A SATURATED CELL. Handed the same gold history, a row that holds the holder map refreshes each object's new holder from the adjacent gold block and scores 1.000 on `give_p1` at every length, so that cell's floor is the ceiling; the pooled teacher-forced floor is 0.7363 at composed@48. The teacher-forced numbers stay what they were — a diagnostic — and the residual they show on `swap_p0` is under the floor of that same class (0.6543 at composed@48).

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


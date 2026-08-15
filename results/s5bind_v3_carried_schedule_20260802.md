# s5_bind_v3 three-cell comparison — from-scratch arm

k=6 · informed chance 1/(k-1) = 0.200 · match · n_eval=1000 · d_model=768 n_layers=8 n_heads=6 batch=16 steps=25000 train_n=80000/stage · supervision=mix

Reading rule pre-registered in `scripts/protocol_s5bind_v3_three_cell_20260731.py`: a cell CLEARS its floor at z > 3.0 AND margin >= 0.15; it FORMS for an architecture on >= 2 of the seeds at every registered length. Per-seed values only — this family is bimodal at the emergence threshold.

**The composed cell has no floor on the GUIDED read, on either channel.** That floor is the one-structure bound `W <= max(k, m) + 1` against the task's `k + m + 1`, and the guided format requires the whole of P then the whole of B at every event — so the k + m slots the bound prices are handed to every policy, the task's own algorithm included, and the class that survives contains the task. It is a property of the PROTOCOL and not of the read: the guided decode accumulates the generated checkpoints into the same context the answer token comes out of. Guided composed cells are reported UNFLOORABLE with the pad reach — what the excluded both-maps class scores on the exact items — beside them. **The previous `gdp_hybrid / guided: V2_NO_GAP_HERE` was read off that floor and is RETRACTED**; the verdict below is what the rule returns without it. The PLAIN read is unaffected: a streaming model with no scratchpad is the class the bound prices.

## Size (compute-matched: shared d_model and depth; `fprm` is weight-tied)

| arch | params | FLOPs/token |
| --- | --- | --- |
| transformer | 76.4M | 165.28M |

## The recipe

`d_model=768` x `n_layers=8`, `n_heads=6`, batch 16, 25000 steps, 80000 items per stage over the three-stage curriculum (stage1_components -> stage2_add_composed -> stage3_composition), supervision `mix` (answer-masked plain documents plus the specs' own per-event checkpoint documents). Every architecture runs the SAME recipe at the same width and depth, which is this repo's compute-matched convention; per-seed weights are saved to `results/s5bind_v3_carried_schedule_20260802_ckpt`, so an added eval length is a decode and not a retrain.

Two numbers in it are set by the hardware, on the documents this run trains on (the composed cell's checkpoint document is 2540 tokens at L=96, mean 958):

- **head dimension 128, so 6 heads at d768.** `GatedDeltaProduct` at head dimension 192 (4 heads at d768) runs 0.836 s/step against 0.108 s/step at 128, same width, same depth, same batch — a kernel path, not a model property.
- **batch 16.** At d768x8 the longest document slice runs out of memory at batch 24 on a 32 GB card (peak 26.9 GB at 16). This run therefore draws 400k sequences per seed from 160k documents per stage.

| arch | measured s/step | train s/seed |
|---|---|---|
| transformer | 0.136 | 3400 |

## The two pairings, and the step multiplier under each

| composed L | component | work-matched L | x steps | x tokens | token-matched L | x steps | x tokens |
|---|---|---|---|---|---|---|---|
| 48 | state | 17 | 4.95x | 3.83x | 80 | 1.10x | 1.04x |
| 48 | bind | 31 | 7.17x | 3.48x | 132 | 2.63x | 1.01x |
| 64 | state | 23 | 4.89x | 3.96x | 108 | 1.08x | 1.02x |
| 64 | bind | 41 | 7.83x | 3.63x | — (unreachable) | — | — |
| 96 | state | 34 | 4.98x | 4.17x | 160 | 1.07x | 1.01x |
| 96 | bind | 62 | 8.96x | 3.73x | — (unreachable) | — | — |

_The work-matched length is the composed stream's own count of that component's events: composed@48 contains 17 swaps and 31 gives. The token-matched length is where that component's forward pass costs what the composed cell costs at L, and it is unreachable on the retrieval component past L=132 — its sampler pins the resolving write into a window that gets exponentially harder to satisfy as the stream grows (protocol.BIND_MATCHED_MAX). That is a property of the instrument, not of this run._

# The depth-matched comparison (pre-registered)

## GUIDED read (n=128)

**composed@48 vs state@17 and bind@31** — carrier chain 5.7 hops on both state legs (composed@48 holds 17 swaps and 31 gives).

| arch | seed | state@17 | composed@48 | bind@31 |
|---|---|---|---|---|
| transformer | 0 | 0.203 | 0.102 | 0.164 |
| transformer | 1 | 0.211 | 0.156 | 0.211 |
| transformer | 2 | 0.133 | 0.250 | 0.156 |
| _floor_ | | 0.219 | unfloorable (pad 0.719) | 0.200 |

## PLAIN read (n=1000)

**composed@48 vs state@17 and bind@31** — carrier chain 5.7 hops on both state legs (composed@48 holds 17 swaps and 31 gives).

| arch | seed | state@17 | composed@48 | bind@31 |
|---|---|---|---|---|
| transformer | 0 | 0.168 | 0.158 | 0.182 |
| transformer | 1 | 0.177 | 0.149 | 0.169 |
| transformer | 2 | 0.149 | 0.181 | 0.147 |
| _floor_ | | 0.200 | 0.204 | 0.200 |

**composed@64 vs state@23 and bind@41** — carrier chain 7.7 hops on both state legs (composed@64 holds 23 swaps and 41 gives).

| arch | seed | state@23 | composed@64 | bind@41 |
|---|---|---|---|---|
| transformer | 0 | 0.179 | 0.155 | 0.176 |
| transformer | 1 | 0.166 | 0.176 | 0.165 |
| transformer | 2 | 0.168 | 0.177 | 0.149 |
| _floor_ | | 0.200 | 0.209 | 0.200 |

**composed@96 vs state@34 and bind@62** — carrier chain 11.3 hops on both state legs (composed@96 holds 34 swaps and 62 gives).

| arch | seed | state@34 | composed@96 | bind@62 |
|---|---|---|---|---|
| transformer | 0 | 0.162 | 0.174 | 0.167 |
| transformer | 1 | 0.151 | 0.161 | 0.180 |
| transformer | 2 | 0.156 | 0.167 | 0.157 |
| _floor_ | | 0.200 | 0.209 | 0.200 |

_A **bold** cell clears its own recomputed floor under the pre-registered rule. Every column of a row costs the same amount of that column's own work; the TOKEN-matched pairing (state@80, bind@132 against composed@48) is the matched-COST control and is in the tables below._

# PLAIN read — answer off the plain prompt, no scratchpad (n=1000)

## state cell — `s5_bind_local_v3_state`

| arch | L16 per-seed | L17 per-seed | L23 per-seed | L34 per-seed | L48 per-seed | L80 per-seed | L108 per-seed | L128 per-seed | L160 per-seed |
|---|---|---|---|---|---|---|---|---|---|
| transformer | 0.172 0.149 0.164 | 0.168 0.177 0.149 | 0.179 0.166 0.168 | 0.162 0.151 0.156 | 0.180 0.167 0.177 | 0.164 0.181 0.175 | 0.144 0.163 0.164 | 0.186 0.159 0.177 | 0.172 0.148 0.164 |
| _floor_ | 0.200 (1.00x) | 0.200 (1.00x) | 0.200 (1.00x) | 0.200 (1.00x) | 0.208 (1.04x) | 0.207 (1.03x) | 0.200 (1.00x) | 0.216 (1.08x) | 0.201 (1.00x) |

## bind cell — `s5_bind_local_v3_bind`

| arch | L16 per-seed | L31 per-seed | L41 per-seed | L62 per-seed | L132 per-seed |
|---|---|---|---|---|---|
| transformer | 0.166 0.153 0.167 | 0.182 0.169 0.147 | 0.176 0.165 0.149 | 0.167 0.180 0.157 | 0.164 0.162 0.163 |
| _floor_ | 0.200 (1.00x) | 0.200 (1.00x) | 0.200 (1.00x) | 0.200 (1.00x) | 0.200 (1.00x) |

## composed cell — `s5_bind_local_v3`

| arch | L16 per-seed | L48 per-seed | L64 per-seed | L96 per-seed |
|---|---|---|---|---|
| transformer | 0.179 0.184 0.173 | 0.158 0.149 0.181 | 0.155 0.176 0.177 | 0.174 0.161 0.167 |
| _floor_ | 0.521 (2.61x) | 0.204 (1.02x) | 0.209 (1.05x) | 0.209 (1.04x) |

# GUIDED read — events forced, checkpoints and answer generated (n=128)

## state cell — `s5_bind_local_v3_state`

| arch | L17 per-seed | L80 per-seed |
|---|---|---|
| transformer | 0.203 0.211 0.133 | 0.133 0.156 0.203 |
| _floor_ | 0.219 (1.09x) | 0.250 (1.25x) |

## bind cell — `s5_bind_local_v3_bind`

| arch | L31 per-seed | L132 per-seed |
|---|---|---|
| transformer | 0.164 0.211 0.156 | 0.109 0.141 0.242 |
| _floor_ | 0.200 (1.00x) | 0.200 (1.00x) |

## composed cell — `s5_bind_local_v3`

| arch | L48 per-seed |
|---|---|
| transformer | 0.102 0.156 0.250 |
| _floor_ | unfloorable (pad 0.719) |

## Guided checkpoint accuracy (per-slot, diagnostic — not the metric)

Read against the copy-the-previous-checkpoint reference, not against 1/k; both are in _What the checkpoint diagnostic says_ below.

| arch | cell | L | per-seed |
|---|---|---|---|
| transformer | state | 17 | 0.190 0.164 0.164 |
| transformer | state | 80 | 0.169 0.162 0.143 |
| transformer | bind | 31 | 0.173 0.402 0.466 |
| transformer | bind | 132 | 0.180 0.405 0.475 |
| transformer | composed | 48 | 0.165 0.166 0.176 |

# Verdict

**transformer / guided: V5_HARNESS_NULL** — no component clears on this read's control grid {'state@17': 0, 'bind@31': 0} (0/2 seeds at best, on state@17). Nothing downstream is interpretable; the next move is the training recipe, not the instrument.

seeds clearing: {'state': {17: 0}, 'bind': {31: 0}, 'composed': {48: None}} (a `None` is a length with no floor on this protocol, not a length where no seed cleared); positive control (some component clears on this read's grid) {'state@17': 0, 'bind@31': 0} of 3 seeds, required ['state@17', 'bind@31']; matched-cost control: {'state': False, 'bind': False} (measured: {'state': True, 'bind': True}); composed pad reach: 0.719; lengths read: {'state': [17], 'bind': [31], 'composed': [48]}

**transformer / plain: V5_HARNESS_NULL** — no component clears on this read's control grid {'state@16': 0, 'bind@16': 0} (0/2 seeds at best, on state@16). Nothing downstream is interpretable; the next move is the training recipe, not the instrument.

seeds clearing: {'state': {17: 0, 23: 0, 34: 0}, 'bind': {31: 0, 41: 0, 62: 0}, 'composed': {48: 0, 64: 0, 96: 0}}; positive control (some component clears on this read's grid) {'state@16': 0, 'bind@16': 0} of 3 seeds, required ['state@16', 'bind@16']; matched-cost control: {'state': False, 'bind': False} (measured: {'state': True, 'bind': True}); lengths read: {'state': [17, 23, 34], 'bind': [31, 41, 62], 'composed': [48, 64, 96]}


# Post-hoc (not pre-registered)

## The composed cell and its depth-matched components, seed by seed (GUIDED)

Each cell at the length carrying the same amount of its own work: composed@48, state@17, bind@31. A blank is a cell this read did not cover.

| arch | seed | state@17 | composed@48 | bind@31 |
|---|---|---|---|---|
| transformer | 0 | 0.203 | 0.102† | 0.164 |
| transformer | 1 | 0.211 | 0.156† | 0.211 |
| transformer | 2 | 0.133 | 0.250† | 0.156 |

A † marks a cell with NO FLOOR on this protocol. The composed cell's floor argument is the one-structure bound, and the guided format writes the whole of P then B at every event — so the k + m slots it prices are handed to every policy, on the answer channel as much as on the trace channel. What the excluded both-maps class reaches on these exact items is 0.719, against a plain-protocol floor of 0.234; it is a lower bound on that class's max and not a bar.

- **transformer**: the composed cell is UNFLOORABLE on this read and its depth-matched state component is at floor on all 3 seeds, so there is nothing to compare it with. The two columns agree because neither cell moved.

## What the checkpoint diagnostic says

Each checkpoint is the whole of P and then the whole of B, k + m = 12 slots per event, and the diagnostic scores every slot of every event. MOST SLOTS DO NOT MOVE: a swap moves 2 of them and a give moves 1, so a model that re-emits its previous checkpoint unchanged at every event is already right on 0.804-0.913 of the slots, against 1/k = 0.167. That copier is the reference, and it is what a per-slot number has to be read against.

| arch | cell | L | per-seed per-slot | copy-the-previous-checkpoint | above the copier |
|---|---|---|---|---|---|
| transformer | bind | 31 | 0.173 0.402 0.466 | 0.901 | -0.728 -0.499 -0.434 |
| transformer | bind | 132 | 0.180 0.405 0.475 | 0.913 | -0.733 -0.508 -0.438 |
| transformer | composed | 48 | 0.165 0.166 0.176 | 0.887 | -0.722 -0.721 -0.711 |
| transformer | state | 17 | 0.190 0.164 0.164 | 0.804 | -0.614 -0.640 -0.640 |
| transformer | state | 80 | 0.169 0.162 0.143 | 0.827 | -0.658 -0.665 -0.684 |

The diagnostic is not a partial trace and no verdict reads it. What IS read is the TRACE read — the final checkpoint's value for the QUERIED slot — a single slot scored against the same gold and against the same floors as this protocol's answer channel (`validity.s5_bind_v3_operative_floor(..., guided=True)`: the component cells floored, the composed cell unfloorable). The copier scores 0.000 on it at every cell because the query gate requires the queried slot to move at least twice and to end different from its stated value.

_A **bold** cell clears its own operative floor under the pre-registered rule; a cell marked `unfloorable` has no floor on that read's protocol and can never be bold, and the `pad` beside it is what the excluded both-maps class scores on the exact items — a lower bound on that class's max, not a bar. Floors are recomputed from that cell's own items and under that read's own protocol: registry rows plus the admitted swept family. The fitted surface ranker is measured beside them (fit 2x2000 / scored 4000 disjoint) and is NOT in any floor — no implementation of it achieves a price the class rule admits. The composed cell's cost multiplier over each component is reported in the pre-registration record in both cost models; the matched-cost lengths in the tables above are the FORWARD-PASS match, which is this regime's cost._
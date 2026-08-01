# s5_bind_v3 three-cell comparison — from-scratch arm

k=6 · informed chance 1/(k-1) = 0.200 · match · n_eval=1000 · d_model=768 n_layers=8 n_heads=6 batch=16 steps=25000 train_n=80000/stage · supervision=mix

Reading rule pre-registered in `scripts/protocol_s5bind_v3_three_cell_20260731.py`: a cell CLEARS its floor at z > 3.0 AND margin >= 0.15; it FORMS for an architecture on >= 2 of the seeds at every registered length. Per-seed values only — this family is bimodal at the emergence threshold.

## Size (compute-matched: shared d_model and depth; `fprm` is weight-tied)

| arch | params | FLOPs/token |
| --- | --- | --- |
| gdp_hybrid | 101.4M | 205.67M |

## The recipe

`d_model=768` x `n_layers=8`, `n_heads=6`, batch 16, 25000 steps, 80000 items per stage over the three-stage curriculum (stage1_components -> stage2_add_composed -> stage3_composition), supervision `mix` (answer-masked plain documents plus the specs' own per-event checkpoint documents).

Two numbers in it are set by the hardware, on the documents this run trains on (the composed cell's checkpoint document is 2540 tokens at L=96, mean 958):

- **head dimension 128, so 6 heads at d768.** `GatedDeltaProduct` at head dimension 192 (4 heads at d768) runs 0.836 s/step against 0.108 s/step at 128, same width, same depth, same batch — a kernel path, not a model property. At 6 heads the GDP state per layer is 6x128x128, larger than the 4x128x128 of the d512x6 run this one supersedes.
- **batch 16.** The flagship recipe's batch of 128 does not fit: at d768x8 the longest document slice runs out of memory at batch 24 on a 32 GB card (peak 26.9 GB at 16). This run therefore draws 400k sequences per seed against the flagship's 3.2M, from 160k documents per stage.

## The gate that ran first (one seed)

One `gdp_hybrid` seed at the flagship width, trained on the STATE cell's own documents alone — the most favourable condition available, no retrieval or composed document competing for the budget — read PLAIN every 2000 steps. 10,000 steps, 30,000 items (60,000 documents), same optimiser and supervision.

| steps | L16 | L48 | L64 | L96 | loss |
|---|---|---|---|---|---|
| 2000 | 0.161 | 0.171 | 0.162 | 0.164 | 0.8904 |
| 4000 | 0.176 | 0.175 | 0.164 | 0.173 | 0.8914 |
| 6000 | 0.186 | 0.170 | 0.153 | 0.168 | 0.8931 |
| 8000 | 0.161 | 0.160 | 0.164 | 0.166 | 0.8940 |
| 10000 | 0.166 | 0.163 | 0.147 | 0.171 | 0.8936 |

Its GUIDED read at L=48 (n=128) is match 0.141 against the 0.208 floor, with per-slot checkpoint accuracy 0.579. The state cell is at floor on both reads at this seed, and the training loss is flat from step 2000 (0.8904) to step 10,000 (0.8936): more steps at this operating point buy nothing.

What one seed settles is one seed. The rule reads >= 2 of 3 seeds because the family is bimodal at the emergence threshold, and the grid below is that bimodality: on the same cell and the same read, seed 0 scores 1.000 and seed 1 scores 0.281. A single floored seed does not separate a recipe that cannot train the state leg from a seed that did not form it.

## The two cost models, and the matched-cost control

| composed L | vs state (steps) | vs state (tokens) | vs bind (steps) | vs bind (tokens) | matched state L | matched bind L |
|---|---|---|---|---|---|---|
| 48 | 1.82x | 1.65x | 5.62x | 2.46x | 80 | 132 |
| 64 | 1.81x | 1.65x | 6.09x | 2.51x | 108 | — (unreachable) |
| 96 | 1.78x | 1.65x | 6.61x | 2.54x | 160 | — (unreachable) |

_A matched length is where that component's own forward pass costs what the composed cell costs at L. Unreachable on the retrieval component past L=132: its sampler pins the resolving write into a window that gets exponentially harder to satisfy as the stream grows, so the cell cannot be run long enough to cost what the composed cell costs at L=64 or L=96 (protocol.BIND_MATCHED_MAX). That is a property of the instrument, not of this run._

# PLAIN read — answer off the plain prompt, no scratchpad (n=1000)

## state cell — `s5_bind_local_v3_state`

| arch | L16 per-seed | L48 per-seed | L64 per-seed | L80 per-seed | L96 per-seed | L108 per-seed | L160 per-seed |
|---|---|---|---|---|---|---|---|
| gdp_hybrid | 0.185 0.186 0.190 | 0.193 0.193 0.206 | 0.211 0.209 0.186 | 0.202 0.204 0.191 | 0.190 0.190 0.173 | 0.207 0.206 0.179 | 0.203 0.205 0.195 |
| _floor_ | 0.200 (1.00x) | 0.208 (1.04x) | 0.200 (1.00x) | 0.207 (1.03x) | 0.210 (1.05x) | 0.200 (1.00x) | 0.201 (1.00x) |

## bind cell — `s5_bind_local_v3_bind`

| arch | L16 per-seed | L48 per-seed | L64 per-seed | L96 per-seed | L132 per-seed |
|---|---|---|---|---|---|
| gdp_hybrid | **1.000** **1.000** **1.000** | **1.000** **1.000** **1.000** | **1.000** **1.000** **1.000** | **1.000** **1.000** **1.000** | **1.000** **0.982** **0.998** |
| _floor_ | 0.200 (1.00x) | 0.200 (1.00x) | 0.200 (1.00x) | 0.200 (1.00x) | 0.200 (1.00x) |

## composed cell — `s5_bind_local_v3`

| arch | L16 per-seed | L48 per-seed | L64 per-seed | L96 per-seed |
|---|---|---|---|---|
| gdp_hybrid | 0.249 0.206 0.235 | 0.215 0.184 0.193 | 0.232 0.201 0.222 | 0.231 0.214 0.222 |
| _floor_ | 0.521 (2.61x) | 0.204 (1.02x) | 0.209 (1.05x) | 0.209 (1.04x) |

# GUIDED read — events forced, checkpoints and answer generated (n=128)

## state cell — `s5_bind_local_v3_state`

| arch | L48 per-seed | L80 per-seed |
|---|---|---|
| gdp_hybrid | **1.000** 0.281 **0.992** | **1.000** 0.188 **1.000** |
| _floor_ | 0.208 (1.04x) | 0.207 (1.03x) |

## bind cell — `s5_bind_local_v3_bind`

| arch | L48 per-seed | L132 per-seed |
|---|---|---|
| gdp_hybrid | **1.000** **0.992** **0.992** | **1.000** **1.000** **1.000** |
| _floor_ | 0.200 (1.00x) | 0.200 (1.00x) |

## composed cell — `s5_bind_local_v3`

| arch | L48 per-seed |
|---|---|
| gdp_hybrid | **0.992** 0.242 **1.000** |
| _floor_ | 0.204 (1.02x) |

## Guided checkpoint accuracy (per-slot, diagnostic — not the metric)

| arch | cell | L | per-seed | per-slot chance |
|---|---|---|---|---|
| gdp_hybrid | state | 48 | 0.573 0.590 0.578 | 0.167 |
| gdp_hybrid | state | 80 | 0.574 0.593 0.583 | 0.167 |
| gdp_hybrid | bind | 48 | 0.593 0.610 0.603 | 0.167 |
| gdp_hybrid | bind | 132 | 0.602 0.611 0.598 | 0.167 |
| gdp_hybrid | composed | 48 | 0.998 0.932 1.000 | 0.167 |

# Verdict

**gdp_hybrid / guided: V2_NO_GAP_HERE** — the composed cell forms wherever both components do. Composition is not a separate difficulty at k=6 / L<=96 in this regime; the lengths or k must move before the cell is worth buying on the frontier.

seeds clearing: {'state': {48: 2}, 'bind': {48: 3}, 'composed': {48: 2}}; positive control (some component clears on this read's grid) {'state@48': 2, 'bind@48': 3} of 3 seeds, required ['state@48', 'bind@48']; matched-cost control: {'state': True, 'bind': True} (measured: {'state': True, 'bind': True}); lengths read: [48]

**gdp_hybrid / plain: V4_COMPONENT_UNREADABLE** — component(s) ['state'] do not form at their own registered lengths {'state': {48: 0, 64: 0, 96: 0}}, while the other one does. A composed failure would be explained by the component that failed, so no composition claim is available — and the dissociation between the components is the result.

seeds clearing: {'state': {48: 0, 64: 0, 96: 0}, 'bind': {48: 3, 64: 3, 96: 3}, 'composed': {48: 0, 64: 0, 96: 0}}; positive control (some component clears on this read's grid) {'state@16': 0, 'bind@16': 3} of 3 seeds, required ['state@16', 'bind@16']; matched-cost control: {'state': False, 'bind': True} (measured: {'state': True, 'bind': True}); lengths read: [48, 64, 96]

Both verdicts read each component at the composed cell's own length, which carries three times the state work and 1.5 times the retrieval work of the cell it is compared against: composed@48 contains 17 swaps and 31 gives against a component@48's 48 of its own kind. The WORK-MATCHED pairing that removes that — state@17/23/34 and bind@31/41/62 against composed@48/64/96 — is registered in `scripts/protocol_s5bind_v3_three_cell_20260731.py` and was not run here.


# Post-hoc (not pre-registered)

## The composed cell and the state component move together, seed by seed (GUIDED, L=48)

| seed | state | composed | bind |
|---|---|---|---|
| 0 | 1.000 (clears) | 0.992 (clears) | 1.000 (clears) |
| 1 | 0.281 | 0.242 | 0.992 (clears) |
| 2 | 0.992 (clears) | 1.000 (clears) | 0.992 (clears) |

The composed cell clears on EXACTLY the seeds where the state component clears, and fails on the seed where it fails, while the retrieval component clears on all three. The pre-registered rule counts seeds per cell and so cannot see this: it is the same statement as the verdict at one operating point, and a stronger one, because it says the composed cell costs this architecture nothing beyond the state leg rather than that both happened to reach 2 of 3.

## What the checkpoint diagnostic says

Each checkpoint is the whole of P and then the whole of B, k + m = 12 slots per event. On a COMPONENT cell one of those halves never moves — the state cell has no gives, the retrieval cell no swaps — and it is also never STATED: the state cell's prompt contains 0 `belongs to` lines and the retrieval cell's 0 `points to` lines, and that constant half takes 38 distinct values across 40 items. It is unknowable from the prompt, so a model that has the queried map exactly and guesses the other half scores (6 x 1.00 + 6 x 0.167) / 12 = 0.583 — which is where every component cell sits (0.573-0.611). The per-slot figure on those cells is therefore ~1.00 on the map the answer comes out of plus ~0.17 on one that cannot be read, so the trace DOES carry those answers; on the composed cell, where both maps are stated and both move, it is 0.932-1.000. No verdict reads it.

_A **bold** cell clears its own operative floor under the pre-registered rule. Floors are recomputed from that cell's own items: registry rows plus the admitted swept family. The fitted surface ranker is measured beside them (fit 2x2000 / scored 4000 disjoint) and is NOT in any floor — no implementation of it achieves a price the class rule admits. The composed cell's cost multiplier over each component is reported in the pre-registration record in both cost models; the matched-cost lengths in the tables above are the FORWARD-PASS match, which is this regime's cost._
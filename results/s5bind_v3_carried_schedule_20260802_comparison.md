# The transformer null: schedule restarted per stage vs carried across stages

`transformer` · d_model 768 · 8 layers · 25000 steps · batch 16 · lr 0.001 · 80000 items per stage · seeds [0, 1, 2] · k=6 · informed chance 0.200 · match.

**CONTROL** (`s5bind_v3_three_cell_depthmatched_20260801.json`): three `train.run` calls, one per stage. Each builds a fresh AdamW and restarts warmup+cosine at lr, so the learning rate goes 0 → lr → 0 three times and the Adam moments are discarded twice.

**CARRIED** (`s5bind_v3_carried_schedule_20260802.json`): one optimizer and one warmup+cosine over the global 25000 steps. Same documents in the same order (they are cached and deterministic), same mixes, same step shares, same seeds, same eval grid, same floors.

A **bold** cell clears its own floor under the pre-registered rule (z > 3.0, margin >= 0.15); a † marks a cell with no floor on that read. Floors are the registered run's own, so both arms are read against identical numbers. Per seed, never a mean — this family is bimodal at the emergence threshold.

## What the two schedules actually are

The mean learning rate over the whole run is the same to four figures under both, so the treatment is not more learning rate — it is the shape.

| stage | steps | control mean lr | control lr at stage start / end | carried mean lr | carried lr at stage start / end |
|---|---|---|---|---|---|
| stage1_components | 8750 | 5.001e-04 | 1.00e-06 / 4.11e-11 | 8.708e-04 | 1.00e-06 / 7.64e-04 |
| stage2_add_composed | 7500 | 5.001e-04 | 1.00e-06 / 5.84e-11 | 5.314e-04 | 7.64e-04 / 2.94e-04 |
| stage3_composition | 8750 | 5.001e-04 | 1.00e-06 / 4.11e-11 | 1.024e-04 | 2.94e-04 / 4.28e-12 |
| _whole run_ | 25000 | 5.0012e-04 | | 5.0004e-04 | |

The control returns the learning rate to its peak 3 times and anneals it to ~0 three times; the carried schedule peaks once. The stage that carries the composition — stage 3, mix 0.70 composed — runs a full 0 -> lr -> 0 cycle under the control at mean 5.001e-04 against the decaying tail's 1.024e-04 under the carried one. The Adam moments entering that stage are ZERO under the control — the optimizer is rebuilt — and carry 16250 steps of history under the carried schedule.

## Verdict, by the pre-registered rule

| read | control | carried |
|---|---|---|
| plain | V5_HARNESS_NULL | V5_HARNESS_NULL |
| guided | V5_HARNESS_NULL | V5_HARNESS_NULL |

## The GUIDED read, ANSWER channel

| cell | seed | control | carried | difference | p |
|---|---|---|---|---|---|
| state@17 | 0 | 0.172 | 0.203 | +0.031 | 0.63 |
| state@17 | 1 | 0.133 | 0.211 | +0.078 | 0.14 |
| state@17 | 2 | 0.211 | 0.133 | -0.078 | 0.14 |
| state@17 | _floor_ | 0.219 | | | |
| state@80 | 0 | 0.164 | 0.133 | -0.031 | 0.6 |
| state@80 | 1 | 0.125 | 0.156 | +0.031 | 0.59 |
| state@80 | 2 | 0.180 | 0.203 | +0.023 | 0.75 |
| state@80 | _floor_ | 0.250 | | | |
| bind@31 | 0 | 0.211 | 0.164 | -0.047 | 0.42 |
| bind@31 | 1 | 0.180 | 0.211 | +0.031 | 0.64 |
| bind@31 | 2 | 0.141 | 0.156 | +0.016 | 0.86 |
| bind@31 | _floor_ | 0.200 | | | |
| bind@132 | 0 | 0.203 | 0.109 | -0.094 | 0.057 |
| bind@132 | 1 | 0.125 | 0.141 | +0.016 | 0.85 |
| bind@132 | 2 | 0.180 | 0.242 | +0.062 | 0.28 |
| bind@132 | _floor_ | 0.200 | | | |
| composed@48 | 0 | 0.125† | 0.102† | -0.023 | 0.69 |
| composed@48 | 1 | 0.156† | 0.156† | +0.000 | 1 |
| composed@48 | 2 | 0.133† | 0.250† | +0.117 | 0.025 |
| composed@48 | _floor_ | unfloorable (pad 0.719) | | | |

## The GUIDED read, TRACE channel

The control's trace column is the ladder decode (`s5bind_v3_trace_ladder_20260801.jsonl`) off the control's OWN checkpoints: the control run predates the trace read and its record carries `trace: null`. It is the same protocol at the same n on the same weights, and every answer value the two records share is bit-identical, so this is that run's trace and not a substitute measurement.

| cell | seed | control | carried | difference | p |
|---|---|---|---|---|---|
| state@17 | 0 | 0.195 | 0.227 | +0.031 | 0.65 |
| state@17 | 1 | 0.195 | 0.117 | -0.078 | 0.12 |
| state@17 | 2 | 0.172 | 0.148 | -0.023 | 0.73 |
| state@17 | _floor_ | 0.219 | | | |
| state@80 | 0 | 0.203 | 0.156 | -0.047 | 0.42 |
| state@80 | 1 | 0.102 | 0.156 | +0.055 | 0.26 |
| state@80 | 2 | 0.203 | 0.164 | -0.039 | 0.52 |
| state@80 | _floor_ | 0.250 | | | |
| bind@31 | 0 | **0.516** | 0.164 | -0.352 | 3.5e-09 |
| bind@31 | 1 | 0.109 | **0.680** | +0.570 | 1.1e-21 |
| bind@31 | 2 | **1.000** | **0.867** | -0.133 | 8.6e-06 |
| bind@31 | _floor_ | 0.200 | | | |
| bind@132 | 0 | **0.414** | 0.102 | -0.312 | 1.1e-08 |
| bind@132 | 1 | 0.148 | 0.344 | +0.195 | 0.00044 |
| bind@132 | 2 | **0.961** | **0.625** | -0.336 | 8e-12 |
| bind@132 | _floor_ | 0.200 | | | |
| composed@48 | 0 | 0.164† | 0.125† | -0.039 | 0.48 |
| composed@48 | 1 | 0.172† | 0.141† | -0.031 | 0.61 |
| composed@48 | 2 | 0.164† | 0.148† | -0.016 | 0.86 |
| composed@48 | _floor_ | unfloorable (pad 0.719) | | | |

## The PLAIN read (answer off the plain prompt, one token, n = 1000)

| cell | seed | control | carried | difference | p |
|---|---|---|---|---|---|
| state@16 | 0 | 0.172 | 0.172 | +0.000 | 1 |
| state@16 | 1 | 0.161 | 0.149 | -0.012 | 0.5 |
| state@16 | 2 | 0.173 | 0.164 | -0.009 | 0.63 |
| state@16 | _floor_ | 0.200 | | | |
| state@17 | 0 | 0.154 | 0.168 | +0.014 | 0.43 |
| state@17 | 1 | 0.147 | 0.177 | +0.030 | 0.078 |
| state@17 | 2 | 0.176 | 0.149 | -0.027 | 0.11 |
| state@17 | _floor_ | 0.200 | | | |
| state@23 | 0 | 0.175 | 0.179 | +0.004 | 0.86 |
| state@23 | 1 | 0.152 | 0.166 | +0.014 | 0.43 |
| state@23 | 2 | 0.181 | 0.168 | -0.013 | 0.48 |
| state@23 | _floor_ | 0.200 | | | |
| state@34 | 0 | 0.159 | 0.162 | +0.003 | 0.9 |
| state@34 | 1 | 0.166 | 0.151 | -0.015 | 0.39 |
| state@34 | 2 | 0.159 | 0.156 | -0.003 | 0.9 |
| state@34 | _floor_ | 0.200 | | | |
| state@80 | 0 | 0.166 | 0.164 | -0.002 | 0.95 |
| state@80 | 1 | 0.174 | 0.181 | +0.007 | 0.73 |
| state@80 | 2 | 0.177 | 0.175 | -0.002 | 0.95 |
| state@80 | _floor_ | 0.207 | | | |
| bind@16 | 0 | 0.153 | 0.166 | +0.013 | 0.46 |
| bind@16 | 1 | 0.166 | 0.153 | -0.013 | 0.46 |
| bind@16 | 2 | 0.144 | 0.167 | +0.023 | 0.17 |
| bind@16 | _floor_ | 0.200 | | | |
| bind@31 | 0 | 0.154 | 0.182 | +0.028 | 0.11 |
| bind@31 | 1 | 0.190 | 0.169 | -0.021 | 0.24 |
| bind@31 | 2 | 0.159 | 0.147 | -0.012 | 0.49 |
| bind@31 | _floor_ | 0.200 | | | |
| bind@41 | 0 | 0.154 | 0.176 | +0.022 | 0.21 |
| bind@41 | 1 | 0.172 | 0.165 | -0.007 | 0.72 |
| bind@41 | 2 | 0.158 | 0.149 | -0.009 | 0.62 |
| bind@41 | _floor_ | 0.200 | | | |
| bind@62 | 0 | 0.184 | 0.167 | -0.017 | 0.35 |
| bind@62 | 1 | 0.163 | 0.180 | +0.017 | 0.34 |
| bind@62 | 2 | 0.162 | 0.157 | -0.005 | 0.81 |
| bind@62 | _floor_ | 0.200 | | | |
| bind@132 | 0 | 0.153 | 0.164 | +0.011 | 0.54 |
| bind@132 | 1 | 0.183 | 0.162 | -0.021 | 0.24 |
| bind@132 | 2 | 0.166 | 0.163 | -0.003 | 0.9 |
| bind@132 | _floor_ | 0.200 | | | |
| composed@16 | 0 | 0.160 | 0.179 | +0.019 | 0.28 |
| composed@16 | 1 | 0.174 | 0.184 | +0.010 | 0.6 |
| composed@16 | 2 | 0.186 | 0.173 | -0.013 | 0.48 |
| composed@16 | _floor_ | 0.521 | | | |
| composed@48 | 0 | 0.160 | 0.158 | -0.002 | 0.95 |
| composed@48 | 1 | 0.150 | 0.149 | -0.001 | 1 |
| composed@48 | 2 | 0.172 | 0.181 | +0.009 | 0.64 |
| composed@48 | _floor_ | 0.204 | | | |
| composed@64 | 0 | 0.162 | 0.155 | -0.007 | 0.71 |
| composed@64 | 1 | 0.162 | 0.176 | +0.014 | 0.44 |
| composed@64 | 2 | 0.176 | 0.177 | +0.001 | 1 |
| composed@64 | _floor_ | 0.209 | | | |
| composed@96 | 0 | 0.195 | 0.174 | -0.021 | 0.25 |
| composed@96 | 1 | 0.152 | 0.161 | +0.009 | 0.62 |
| composed@96 | 2 | 0.168 | 0.167 | -0.001 | 1 |
| composed@96 | _floor_ | 0.209 | | | |

## The teacher-forced probe — moving slots, gold history

The diagnostic that identified the artifact: argmax at every checkpoint slot with the TRUE history in front of the model, restricted to the slots whose value differs from the previous checkpoint's, which is the only part a copier does not get for free. It is not a score on the task — the true history is exactly what the task withholds — and no verdict reads it.

| cell | seed | control | carried | difference |
|---|---|---|---|---|
| state@17 | 0 | 0.575 | 0.549 | -0.026 |
| state@17 | 1 | 0.575 | 0.617 | +0.043 |
| state@17 | 2 | 0.664 | 0.644 | -0.020 |
| state@80 | 0 | 0.588 | 0.566 | -0.022 |
| state@80 | 1 | 0.580 | 0.621 | +0.041 |
| state@80 | 2 | 0.669 | 0.646 | -0.022 |
| bind@31 | 0 | 0.590 | 0.085 | -0.504 |
| bind@31 | 1 | 0.000 | 0.739 | +0.739 |
| bind@31 | 2 | 1.000 | 0.848 | -0.152 |
| bind@62 | 0 | 0.615 | 0.098 | -0.517 |
| bind@62 | 1 | 0.000 | 0.723 | +0.723 |
| bind@62 | 2 | 1.000 | 0.840 | -0.160 |
| composed@48 | 0 | 0.314 | 0.323 | +0.009 |
| composed@48 | 1 | 0.243 | 0.407 | +0.164 |
| composed@48 | 2 | 0.520 | 0.444 | -0.076 |
| composed@96 | 0 | 0.324 | 0.328 | +0.004 |
| composed@96 | 1 | 0.240 | 0.422 | +0.182 |
| composed@96 | 2 | 0.527 | 0.446 | -0.081 |

## Final training loss per stage

The schedule acts on the optimisation, so the loss is the first place a difference would show. Stage 3's loss is on the composition-weighted mix.

| seed | stage | control | carried |
|---|---|---|---|
| 0 | stage1_components | 0.8997 | 0.9224 |
| 0 | stage2_add_composed | 0.8931 | 0.9031 |
| 0 | stage3_composition | 0.3285 | 0.3413 |
| 1 | stage1_components | 0.3974 | 0.4495 |
| 1 | stage2_add_composed | 0.9016 | 0.9001 |
| 1 | stage3_composition | 0.3840 | 0.3451 |
| 2 | stage1_components | 0.8975 | 0.8801 |
| 2 | stage2_add_composed | 0.8933 | 0.8955 |
| 2 | stage3_composition | 0.2878 | 0.3035 |

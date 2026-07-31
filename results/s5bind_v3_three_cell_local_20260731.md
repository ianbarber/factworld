# s5_bind_v3 three-cell comparison — from-scratch arm

k=6 · informed chance 1/(k-1) = 0.200 · match · n_eval=1000 · d_model=512 n_layers=6 batch=32 steps=10000 train_n=14000/stage · supervision=mix

Reading rule pre-registered in `scripts/protocol_s5bind_v3_three_cell_20260731.py`: a cell CLEARS its floor at z > 3.0 AND margin >= 0.15; it FORMS for an architecture on >= 2 of the seeds at every registered length. Per-seed values only — this family is bimodal at the emergence threshold.

## Size (compute-matched: shared d_model and depth; `fprm` is weight-tied)

| arch | params | FLOPs/token |
| --- | --- | --- |
| fprm | 4.8M | 57.78M |
| gdp_hybrid | 33.2M | 68.33M |
| transformer | 25.7M | 57.76M |

# PLAIN read — answer off the plain prompt, no scratchpad (n=1000)

## state cell — `s5_bind_local_v3_state`

| arch | L16 per-seed | L48 per-seed | L64 per-seed | L80 per-seed | L96 per-seed | L108 per-seed | L160 per-seed |
|---|---|---|---|---|---|---|---|
| fprm | 0.172 0.174 0.159 | 0.191 0.166 0.148 | 0.192 0.189 0.175 | 0.165 0.199 0.181 | 0.182 0.183 0.177 | 0.150 0.185 0.170 | 0.163 0.190 0.150 |
| gdp_hybrid | 0.169 0.221 0.177 | 0.176 0.202 0.169 | 0.183 0.201 0.158 | 0.182 0.213 0.176 | 0.169 0.182 0.190 | 0.193 0.201 0.176 | 0.176 0.208 0.167 |
| transformer | 0.151 0.168 0.170 | 0.144 0.143 0.148 | 0.158 0.163 0.163 | 0.187 0.180 0.180 | 0.158 0.171 0.168 | 0.174 0.190 0.155 | 0.152 0.167 0.165 |
| _floor_ | 0.201 (1.00x) | 0.200 (1.00x) | 0.200 (1.00x) | 0.207 (1.03x) | 0.203 (1.02x) | 0.201 (1.00x) | 0.205 (1.03x) |

## bind cell — `s5_bind_local_v3_bind`

| arch | L16 per-seed | L48 per-seed | L64 per-seed | L96 per-seed | L132 per-seed |
|---|---|---|---|---|---|
| fprm | 0.170 0.186 0.298 | 0.163 0.175 0.337 | 0.155 0.150 0.293 | 0.177 0.168 0.270 | 0.162 0.154 0.268 |
| gdp_hybrid | **1.000** **1.000** **1.000** | **1.000** **1.000** **1.000** | **1.000** **1.000** **1.000** | **1.000** **1.000** **1.000** | **1.000** **1.000** **1.000** |
| transformer | 0.170 0.171 0.153 | 0.163 0.180 0.200 | 0.156 0.162 0.172 | 0.136 0.176 0.180 | 0.162 0.162 0.157 |
| _floor_ | 0.200 (1.00x) | 0.200 (1.00x) | 0.200 (1.00x) | 0.200 (1.00x) | 0.200 (1.00x) |

## composed cell — `s5_bind_local_v3`

| arch | L16 per-seed | L48 per-seed | L64 per-seed | L96 per-seed |
|---|---|---|---|---|
| fprm | 0.179 0.194 0.158 | 0.150 0.177 0.170 | 0.186 0.183 0.159 | 0.155 0.204 0.170 |
| gdp_hybrid | 0.197 0.230 0.162 | 0.190 0.217 0.173 | 0.194 0.189 0.167 | 0.185 0.197 0.165 |
| transformer | 0.156 0.161 0.165 | 0.164 0.158 0.187 | 0.188 0.140 0.177 | 0.163 0.172 0.167 |
| _floor_ | 0.509 (2.54x) | 0.232 (1.16x) | 0.243 (1.21x) | 0.232 (1.16x) |

# GUIDED read — events forced, checkpoints and answer generated (n=128)

## state cell — `s5_bind_local_v3_state`

| arch | L48 per-seed |
|---|---|
| fprm | 0.109 0.188 0.133 |
| gdp_hybrid | **0.430** **0.617** 0.211 |
| transformer | 0.242 0.125 0.117 |
| _floor_ | 0.200 (1.00x) |

## bind cell — `s5_bind_local_v3_bind`

| arch | L48 per-seed |
|---|---|
| fprm | 0.172 0.211 0.211 |
| gdp_hybrid | **1.000** **0.992** **1.000** |
| transformer | 0.180 0.172 0.172 |
| _floor_ | 0.200 (1.00x) |

## composed cell — `s5_bind_local_v3`

| arch | L48 per-seed |
|---|---|
| fprm | 0.141 0.203 0.172 |
| gdp_hybrid | **0.492** 0.352 0.234 |
| transformer | 0.156 0.156 0.172 |
| _floor_ | 0.232 (1.16x) |

## Guided checkpoint accuracy (per-slot, diagnostic — not the metric)

| arch | cell | L | per-seed | per-slot chance |
|---|---|---|---|---|
| fprm | state | 48 | 0.196 0.175 0.214 | 0.167 |
| fprm | bind | 48 | 0.553 0.555 0.583 | 0.167 |
| fprm | composed | 48 | 0.176 0.164 0.353 | 0.167 |
| gdp_hybrid | state | 48 | 0.579 0.550 0.556 | 0.167 |
| gdp_hybrid | bind | 48 | 0.589 0.599 0.594 | 0.167 |
| gdp_hybrid | composed | 48 | 0.935 0.679 0.812 | 0.167 |
| transformer | state | 48 | 0.162 0.161 0.167 | 0.167 |
| transformer | bind | 48 | 0.178 0.460 0.521 | 0.167 |
| transformer | composed | 48 | 0.168 0.170 0.169 | 0.167 |

# Verdict

**fprm / guided — pre-registered: V5_HARNESS_NULL** — the state component is at floor at the shortest trained length L=16 (0/2 seeds). The harness is not training this family at this width and budget; no cell downstream is interpretable.

**repaired: V5_HARNESS_NULL** — no component clears anywhere (0 seeds). Nothing downstream is interpretable; the next move is the training recipe, not the instrument.

seeds clearing: {'state': {48: 0}, 'bind': {48: 0}, 'composed': {48: 0}}; L=16 state control: 0/3; any-component control: 0/3; matched-cost control: {'state': None, 'bind': None} (measured: {'state': False, 'bind': False}); lengths read: [48]

**fprm / plain — pre-registered: V5_HARNESS_NULL** — the state component is at floor at the shortest trained length L=16 (0/2 seeds). The harness is not training this family at this width and budget; no cell downstream is interpretable.

**repaired: V5_HARNESS_NULL** — no component clears anywhere (0 seeds). Nothing downstream is interpretable; the next move is the training recipe, not the instrument.

seeds clearing: {'state': {48: 0, 64: 0, 96: 0}, 'bind': {48: 0, 64: 0, 96: 0}, 'composed': {48: 0, 64: 0, 96: 0}}; L=16 state control: 0/3; any-component control: 0/3; matched-cost control: {'state': False, 'bind': False} (measured: {'state': True, 'bind': True}); lengths read: [48, 64, 96]

**gdp_hybrid / guided — pre-registered: V5_HARNESS_NULL** — the state component is at floor at the shortest trained length L=16 (0/2 seeds). The harness is not training this family at this width and budget; no cell downstream is interpretable.

**repaired: V1_UNCONTROLLED** — both components form and the composed cell does not — the V1 pattern — but the matched-cost control is absent for ['state', 'bind'], so 'beyond the step multiplier' is not established. The cells separate; the cause does not.

seeds clearing: {'state': {48: 2}, 'bind': {48: 3}, 'composed': {48: 1}}; L=16 state control: 0/3; any-component control: 3/3; matched-cost control: {'state': None, 'bind': None} (measured: {'state': False, 'bind': False}); lengths read: [48]

**gdp_hybrid / plain — pre-registered: V5_HARNESS_NULL** — the state component is at floor at the shortest trained length L=16 (0/2 seeds). The harness is not training this family at this width and budget; no cell downstream is interpretable.

**repaired: V4_COMPONENT_UNREADABLE** — component(s) ['state'] do not form at their own registered lengths {'state': {48: 0, 64: 0, 96: 0}}, while the other one does. A composed failure would be explained by the component that failed, so no composition claim is available — and the dissociation between the components is the result.

seeds clearing: {'state': {48: 0, 64: 0, 96: 0}, 'bind': {48: 3, 64: 3, 96: 3}, 'composed': {48: 0, 64: 0, 96: 0}}; L=16 state control: 0/3; any-component control: 3/3; matched-cost control: {'state': False, 'bind': True} (measured: {'state': True, 'bind': True}); lengths read: [48, 64, 96]

**transformer / guided — pre-registered: V5_HARNESS_NULL** — the state component is at floor at the shortest trained length L=16 (0/2 seeds). The harness is not training this family at this width and budget; no cell downstream is interpretable.

**repaired: V5_HARNESS_NULL** — no component clears anywhere (0 seeds). Nothing downstream is interpretable; the next move is the training recipe, not the instrument.

seeds clearing: {'state': {48: 0}, 'bind': {48: 0}, 'composed': {48: 0}}; L=16 state control: 0/3; any-component control: 0/3; matched-cost control: {'state': None, 'bind': None} (measured: {'state': False, 'bind': False}); lengths read: [48]

**transformer / plain — pre-registered: V5_HARNESS_NULL** — the state component is at floor at the shortest trained length L=16 (0/2 seeds). The harness is not training this family at this width and budget; no cell downstream is interpretable.

**repaired: V5_HARNESS_NULL** — no component clears anywhere (0 seeds). Nothing downstream is interpretable; the next move is the training recipe, not the instrument.

seeds clearing: {'state': {48: 0, 64: 0, 96: 0}, 'bind': {48: 0, 64: 0, 96: 0}, 'composed': {48: 0, 64: 0, 96: 0}}; L=16 state control: 0/3; any-component control: 0/3; matched-cost control: {'state': False, 'bind': False} (measured: {'state': True, 'bind': True}); lengths read: [48, 64, 96]


_A **bold** cell clears its own operative floor under the pre-registered rule. Floors are recomputed from that cell's own items (registry rows plus the admitted swept family plus the fitted surface ranker, fit 2000 / scored 4000 disjoint). The composed cell's cost multiplier over each component is reported in the pre-registration record in both cost models; the matched-cost lengths in the tables above are the FORWARD-PASS match, which is this regime's cost._
**FactWorld: An Oracle-Validated Instrument for Composing Recall, State-Tracking, and Knowledge**

*Ian Barber — June 2026*

## Abstract

Sequence models for agentic and code-like workloads often need to do more than recall facts or track state in isolation. They must combine these abilities in one forward pass, often beyond the training horizon, with knowledge that may come either from the prompt or from the weights. Existing probes usually test the pieces separately: multi-query associative recall tests lookup, while S5-style word problems test non-commuting state. We introduce **FactWorld**, a synthetic instrument for measuring their composition.

FactWorld is a deterministic world of agents, objects, roles, and facts. Every item is rendered from a knowledge base, and every label is computed by a symbolic oracle, correct by construction. The benchmark crosses recall with easy and hard state-tracking, and separates in-context from parametric facts. A single query can require resolving state first, then recalling a property of the resolved entity.

We validate the instrument by reproducing known single-capability dissociations: product recurrences extrapolate non-abelian state where shortcut-learning mixers collapse, and attention solves canonical one-hop associative recall. The new measurement is the composition gap. At small scale, binding × in-context recall fails unless the recall leg degenerates to memorization. At ~45M parameters, matched learning-rate sweeps show that gated-delta recurrent hybrids learn the composite where a matched transformer does not. Within the recurrent family, the Householder-product variant has a broader convergent learning-rate band and also supplies the non-abelian state leg. Decomposing the composite shows it is learned as a resolve-then-recall pipeline rather than two independent legs: the model recalls the property of whatever agent it resolves, so recall is routed through the binding. FactWorld exposes where isolated capabilities compose, where they fail, and which architectural lever moves the gap.

**Code & data.** The instrument, the frozen task suite, and the per-claim experiment scripts are at [github.com/ianbarber/factworld](https://github.com/ianbarber/factworld); every headline number maps to one script in `scripts/` and a table in `docs/`, with  reproduction commands in the README. The data, oracle, and evaluation layer is pure-stdlib; the training runs require a CUDA GPU.

## 1. Introduction

Two capabilities that sub-quadratic sequence models must learn for length extrapolation sit on opposite sides of the architecture problem. Non-abelian state-tracking, probed by the S5 word problem (Liu et al., 2023), composes a non-commuting group action over a sequence. Long-range associative recall, probed by MQAR (Arora et al., 2023), is content-addressed lookup over a growing context.

Each has a known recipe. For state-tracking, a non-diagonal per-token transition with negative eigenvalues (Grazzi et al., 2024), realized as a product of Householder reflections (Siems et al., 2025). For recall, one or a few soft-attention layers over a recurrent backbone (Lieber et al., 2024; Wang et al., 2025). That hybrid ratio first appeared in complementary state-space stacks (Fu et al., 2022).

Agentic and code-like workloads demand their *composition*. A single query must track state *and* recall, past the training horizon, with knowledge supplied in-context or stored in weights. We build an instrument to ask that directly.

Prior work establishes each leg in isolation. State-tracking expressivity hits a TC⁰/NC¹ ceiling (Liu et al., 2023; Merrill & Sabharwal, 2023; Merrill, Petty & Sabharwal, 2024; Barrington, 1989). A non-diagonal mechanism escapes it (Grazzi et al., 2024; Siems et al., 2025). Associative recall and the attention-rescue recipe are well documented (Arora et al., 2023; Lieber et al., 2024; Wang et al., 2025). The optimization-vs-capacity account of recall failure is argued by Okpekpe & Orvieto (2025). A separate line asks where parametric knowledge lives in weights (Geva et al., 2021; Meng et al., 2022). FactWorld measures their composition, separates parametric from in-context fact sources, and scales difficulty to test when composition becomes learnable.

We validate the instrument before composing. On its single-capability cells it reproduces the established dissociations: product-recurrence extrapolation of S5/A5, transformer shortcut-brittleness, and attention solving 1-hop MQAR. These replications confirm that the instrument transfers the field's concepts.

**Contributions.**

1. **FactWorld, an oracle-validated instrument for capability composition.** Its properties are: (a) labels correct by construction. Every answer is computed by a symbolic oracle and is never parsed from rendered text, so labels cannot leak; the renderer produces only the prompt. (b) A validity gate plus an adversarial leakage suite certifies that no shallow baseline clears the floor. (c) A crossed design: recall × {easy, hard}-state × {parametric, in-context} on one world with matched answer distributions. (d) A composition primitive: a query that resolves a stateful binding *and then* recalls a property.

2. **A reusable, runnable benchmark** (§2.1). We freeze the instrument into a versioned task suite (`factworld/tasks.py`, suite v1.0) with one-command execution (`scripts/run_benchmark.py`). It includes scored tasks and labelled controls. Train and OOD-length splits are deterministic, labels resolve through the oracle, and the canonical metric is position-strict exact match.

3. **The composition gap.** Decomposing the composite shows it is learned as a resolve-then-recall pipeline, not two independent legs: on holder-wrong examples the model recalls the property of the agent it actually resolved (routing ≈ 1.0 at the train length), so recall is conditioned on the binding (§4). At a ~45M-param, matched-compute point, gated-delta recurrent hybrids cross the composite where a param-matched transformer does not. Within the recurrent family, the product-structured GatedDeltaProduct hybrid has a markedly broader convergent learning-rate band than the single-delta variant and additionally supplies the non-abelian state leg. The state and recall mechanisms are prior art (Grazzi et al., 2024; Siems et al., 2025).

## 2. The instrument

FactWorld is a small, fully-known synthetic world of *agents* (`g0, g1, …`), *objects* (`o0, o1, …`), and a static *fact* per agent. An opaque attribute maps to a value, e.g. `g3`'s `a0` is `v42`. The value's spelling carries no signal. Events unfold over time. Objects are **given** to agents (`give o3 to g1`). In harder tasks agents **swap roles**. The world generates from a seed. A symbolic **oracle** tracks the exact state after every event. The true answer to any question is known by construction. Labels never read back out of the text.

Questions can require **combining** abilities in one answer:

- *Recall* — "what is `a0` of `g3`?" (look up a fact).
- *State-tracking* — "who holds `o3`?" (replay the give-history, last-write-wins).
- *Composition* — "what is `a0` of **the holder of** `o3`?" — resolve who holds `o3` (state-tracking) *then* recall that agent's `a0` (recall), in one step.

The fact a question needs is either **in-context** (the facts appear in the prompt) or **parametric/in-weights** (the facts are fixed across all training, so a model can memorize them). This separates "reading from context" from "knowing from training." Event histories can be made longer at test time than in training. We can then tell whether a model learned the rule or a length-specific shortcut.

**A knowledge base as the single source of truth.** A `World` is deterministic from a seed. It holds a static layer (recall attributes, initial object holders, an initial Sₖ role assignment) and samples event chains on demand. Documents and evaluation items render *from* the KB through a deterministic template renderer. The render↔parse round-trip is enforced. Identifiers are atomic, closed-vocabulary tokens (`e17`, `a3`, `v42`, `o2`, `g4`, `r0`). The value pool is shared and opaque. No token reveals its type or answer. Tokenization is exact whitespace splitting.

**Three measured axes.** *Recall* — `(entity, attribute) → value`. *Easy-state* — an object's holder after a `move`/`give` chain (last-write-wins; bounded state). *Hard-state* — k agents permuting k roles via `swap`/`cycle` (a word problem over the symmetric group Sₖ).

**The validity gate.** The instrument is only trustworthy if its tasks cannot be solved shallowly. The gate certifies that the oracle scores 100%. Majority, recency, first-position, naive-Bayes, and object-blind-recency probes all sit near floor, and answer distributions are balanced (excess-KL over the finite-sample bias). Hard-state in particular admits no shallow shortcut. The gate forced several design choices: identity initial role assignment (a random hidden initial would be unidentifiable from history-only data), touched-object easy queries, disjoint-namespace auxiliary worlds, a shared opaque value vocabulary, and cycles of length ≥ 3.

### 2.1 A runnable benchmark

Five scored tasks (the `REPORTED` set) span the axes the instrument separates: recall, state, their composition, the parametric-vs-in-context source of a fact, and composition depth:

- `recall_copy_v1` — 1-of-N in-context-copy recall. The distractor pool grows by `length`, a real recall-extrapolation axis.
- `binding_v1` — last-write-wins state. This *is* the delta-rule update, so delta-rule recurrences have a structural prior here.
- `composite_copy_v1` — binding × in-context-copy recall, the 2-hop composition primitive and the task carried to scale in §5. It is bimodal at threshold, so we report it as p(converge) over seeds.
- `conflict_v1` — the model first memorizes a fixed agent→value map (reinforced in training), then must *override* it from a contradicting in-context fact. The gold answer is the in-context value. A weight-defaulting model answers the memorized one.
- `chain_v1` — composition depth: a depth-*k* pointer chase (`length` is the chain depth). Depths are kept below *k* so no recency shortcut exists.

Each task is a frozen `TaskSpec` exposing the difficulty axes: agents/roles `k`, recall pool, working-set size, value-vocabulary, and separate `train_lengths` / held-out `eval_lengths`. `spec.scaled(...)` returns a harder variant. The RNG keys on `(spec, split, length, index)`. Train and every OOD-length test split are fixed and provably disjoint. The metric is `score_exact`: position-strict exact match of the full answer span. Labelled controls (memorized fixed-map versions) and experimental tasks (e.g. non-abelian Sₖ) carry a `kind` field and are excluded from the headline set.

## 3. Single-capability baselines

We replicate: (i) transformer shortcut-brittleness on S5 (Liu et al., 2023); (ii) the single-delta / non-negative-eigenvalue GatedDeltaNet [0,1] failure (Grazzi et al., 2024); (iii) the Householder-product extrapolation (Siems et al., 2025); and (iv) a GRU NC¹ positive control. 

### 3.1 State-tracking

A non-diagonal per-token transition with negative eigenvalues unlocks non-abelian state-tracking (Grazzi et al., 2024). Realizing it as a per-token product of n_h Householder reflections solves the S5 word problem once n_h ≥ 4 (DeltaProduct; Siems et al., 2025). Parameter-matched controls show the load-bearing variable is the ≥ 4 non-commuting reflections, not capacity. Mechanistically, such a transition composes as a non-commutative prefix-product (NC¹), escaping the commutative scan (TC⁰) of a diagonal transition (Merrill & Sabharwal, 2023; Merrill, Petty & Sabharwal, 2024; Barrington, 1989).

Our backbone adopts this structure. We take the production recall hybrid (recurrent blocks with one RoPE soft-attention layer per four, `attention_ratio` 0.25) and flip only two transition flags: `num_householder` 1→4 and `allow_neg_eigval` False→True. This swaps the shortcut-learning GatedDeltaNet backbone (Yang, Kautz & Hatamizadeh, 2024) for the S5-capable GatedDeltaProduct one.

We reproduce this mechanism in our own repo on the canonical group word problem under dense per-token supervision. The model predicts the running prefix product at every position. We train at length 32 and evaluate to 4×, across 3 seeds. 

The attention-free product backbone (`gdp_pure`, n_h = 4, neg-eig) and a GRU reference extrapolate S5. Per-token accuracy is ≈ 0.99 and 1.00 at 4×. The single-delta / non-negative-eigenvalue GatedDeltaNet and the softmax transformer shortcut-learn. They fit the train length, then collapse to exact 0 at 2×/4×. The same dissociation holds on A5, the smallest non-abelian simple group, so the result is not S5-specific.

Because the probe is attention-free, it attributes the state leg to the product recurrence. The same backbone also supplies the recall leg attention-free (§3.2).

We do not claim a clean reflection-count necessity from this isolated probe. The single-reflection n_h = 1 null is seed-fragile here (it extrapolates on 2/3 seeds): over ≥ 4 tokens a product of single reflections can compose a 5-cycle (Cartan–Dieudonné), so this isolated probe cannot separate optimization from expressivity. The load-bearing mechanism-not-capacity evidence sits on the memorized-map composite at fixed parameters and the default recipe (§4).

### 3.2 Recall

On canonical (1-hop) MQAR, a single soft-attention layer over a recurrent stack solves recall. This is the attention-rescue result of Zoology (Arora et al., 2023) and the hybrid-ratio analysis of Wang et al. (2025). Okpekpe & Orvieto (2025) caution that apparent pure-recurrent failures often reflect optimization, not a capability limit.

We adopt that hybrid (one RoPE attention layer over the recurrent backbone) and verify recall in isolation on FactWorld. We use a resampled agent→value map so nothing can be memorized; the task is pure in-context copy. On this 1-hop read-out the instrument reproduces the field's result exactly: a bare transformer solves recall perfectly and robustly (1.00 at 4 and 8 heads, both seeds).

The instrument isolates 1-hop recall from a *read-out-deferred* regime. Our recall query requests the value at a separated answer position, not as the next token after the key, because composition (§4) requires reading a bound value at an arbitrary later point.

Four architectures, 3 seeds, at a small distractor pool:

| architecture | in-context recall (pool 2) |
|---|---|
| **gdp_hybrid** | **1.00 ± 0.00** |
| gdn_hybrid | 0.51 ± 0.02 |
| transformer | 0.48 ± 0.02 |
| gru | 0.33 ± 0.22 |

Only the product hybrid does this deferred recall robustly. The gdn hybrid has the same RoPE attention layer yet only matches the transformer (0.51). One attention layer does not rescue recall in this deferred regime.

The same transformer that solves 1-hop robustly is seed-fragile on the deferred read-out (0.64 at heads 8; the table's 0.48 is the heads-4 baseline). The attention-free product backbone solves the identical deferred task robustly (gdp_pure 1.00, both seeds). Attention aces 1-hop MQAR, but the deferred read-out is free for a recurrence that carries the binding in state; for attention it is only fragilely learnable at this budget.

*Attribution (attention-free).* Removing attention entirely shows that the product structure is sufficient for robust deferred recall at this scale. `gdp_pure` tracks across pools (0.92 / 0.35 at pool 4 / 8), while the non-negative-eigenvalue single-delta `gdn_pure` floors (0.61 → 0.12). The extra Householder reflections are sufficient for robust deferred recall at this scale, mirroring their role for state (§3.1). The attention layer is not required for either leg in isolation; composing the two in one query is the open problem (§4–§5).

Recall also has its own difficulty axis, the distractor pool. gdp degrades smoothly as the pool grows (1.00 → 0.65 → 0.33 at pool 2 / 6 / 8). Recall is "easy in isolation" only at modest binding load.

## 4. The composition gap

Each leg is solvable in isolation (§3). Composing them in one query is the hard part. The four-architecture baseline under our default recipe shows a clean dissociation:

- **Recall** (deferred read-out) is solved robustly only by the product backbone (gdp 1.00). Attention is seed-fragile in this regime, though it aces 1-hop MQAR. The attention-free contrast—gdp_pure 1.00 vs. gdn_pure 0.61—shows the product structure, not the attention pathway, is the reliable path to deferred recall in this setting (§3.2).
- **Last-write-wins binding** is led by the two recurrent hybrids. gdp scores 0.92–1.00 and gdn 0.79–0.91 in-distribution. Both sit at ≈0.5 at 4× (high-variance/bimodal: gdp 0.54 ± 0.32, gdn 0.48 ± 0.22 at L64), well above transformer (0.30) and gru (0.25). Last-write-wins is the delta-rule update (Yang, Kautz & Hatamizadeh, 2024), so the delta-rule recurrences carry a structural prior here. gdp is at least as strong as gdn (the two are within seed noise at 4×).
- **The pointer-chain** (`chain_v1`) is learned in-distribution only by the two recurrent hybrids. Transformer and gru floor throughout. The hybrids themselves collapse one hop past the training depth.
- **The two-hop composite** (`composite_copy_v1`) floors for all four under the default recipe.

The recurrent hybrids have the structural prior for binding, and the product structure is sufficient for recall in the deferred-readout regime — but composing the two in one query is the open problem. The genuine composite floors for every architecture at small scale (~6M); §5 reaches it only at ~45M with the product hybrid (seed- and recipe-fragile).

At small scale, composition holds up at length only when the recall leg degenerates to a memorized lookup. With the agent→value map fixed (memorizable), the gdp composite is solvable and partially extrapolates (0.79 at L16 → 0.35 at L64). Resampled per example (genuine in-context copy), it floors for all four architectures (≤ 0.02 at the benchmark baseline). Binding × parametric recall likewise floors at length for every architecture.

**The composite is learned as a resolve-then-recall pipeline, not two independent legs.** Decomposing the genuine composite into its state leg (holder resolved?) and recall leg (value correct?) shows recall is routed through the binding. It is learned all-or-nothing: 3/5 seeds form the circuit; 2/5 floor with the recall leg never forming even when the binding is intact (a floored seed reaches holder 0.47 at value 0.02), locating the bottleneck at recall-under-composition. 

On the converged seeds the model essentially never recalls the correct value without first resolving the binding (P(value | holder wrong) ≈ 0), and the token-level reason is genuine routing, not breakdown: on holder-wrong examples it emits the property of *the agent it actually resolved* 97–100% of the time at the train length, and at that length never falls silent — it executes resolve-then-recall and recalls the wrong agent's value. Recall is therefore conditioned on the binding, not an independent parallel leg. At 4× the pipeline degrades gracefully with the binding (routing 0.53–0.91) but keeps its structure. This is all-or-nothing *optimization* (the circuit forms or it does not).

## 5. Scale and the architecture lever

§4 characterizes the composition gap at our small compute-matched scale. The instrument's scalable difficulty axis lets us ask whether the gap is a capability limit or a small-model artifact. 

We hold the task fixed: binding × genuine in-context-copy recall, with the map resampled per example so the recall leg cannot be memorized. We use a small pool (k = 5) so that leg is independently learnable. Any composite floor is therefore attributable to composition, not recall capacity. This is the k = 5, recall-pool-5, value-vocab-64 composite, scored by `iso.strict_eval` (both the resolved holder and the recalled value required, harvested from the generated span — more lenient on position than the benchmark's `score_exact`; §6).

At a matched ~45M-param, matched-compute budget, 25k steps, with p(converge) = the fraction of seeds clearing 0.5 (gdp 5 seeds, transformer 5, gdn 3). This **default-recipe** table floors; the favorable result is recipe-sensitive and quantified in §5.2 (tuned-LR 5-seed: gdp L16 0.87, 5/5; L64 0.76, 3/5):

| arm (~45M, matched compute) | L16 (in-distribution) | L64 (4×) |
|---|---|---|
| **gdp_hybrid** (n_h = 4 Householder) | **0.48 ± 0.21  (1/5)** | 0.17 ± 0.09  (0/5) |
| transformer (param-matched) | 0.01 ± 0.01  (0/5) | 0.00 ± 0.00  (0/5) |
| gdn_hybrid (no product structure) | 0.01 ± 0.00  (0/3) | 0.01 ± 0.00  (0/3) |

At the default recipe, the composite floors for all four architectures. Only gdp shows any lift, and it is seed-fragile (1/5). A 6M gdp hybrid floors the same composite (0.08 / 0.02). The ≈45M step lifts it off the floor. A pure-copy positive control passes (1.00) at 45M. The floor is composition, not retrieval.

A matched learning-rate sweep asks whether the gap is optimization or capacity. We sweep five rates across the recurrent hybrids and the transformer (2 seeds per cell). The results read in three layers.

### 5.1 Layer 1: Recurrence versus attention

The param-matched transformer floors across its entire learning-rate sweep (0/10). Both gated-delta recurrent hybrids have at least one learning rate where they converge. This is the recurrent-backbone-versus-attention contrast. The floor is not a head-count or init artifact: a fair attention recipe — n_heads = 8 (head_dim 64) with residual-scaled init — still floors 0/10 (best run 0.20).

### 5.2 Layer 2: Product versus single-delta

Within the recurrent family, the product structure is associated with a markedly broader convergent band. gdp's in-distribution accuracy stays high across 3e-4–2e-3 (7/10 cells; per-LR means 0.88 / 0.99 / 0.87 / 0.41). The single-delta GatedDeltaNet converges robustly only at lr 3e-4 (5 seeds there: L16 0.78 ± 0.27, 4/5) and floors at every other rate; its 0/3 at the default lr 1e-3 was therefore learning-rate-specific, not a capability bound.

A five-seed confirmation at gdp's best rate (lr 5e-4) pins the tuned estimate. In-distribution convergence is robust (L16 0.87 ± 0.15, 5/5). Length-extrapolation holds on a majority of seeds (L64 0.76 ± 0.25, 3/5). The default recipe (lr 1e-3) floored extrapolation (0/5 at 4×), so the seed-fragility was largely a recipe artifact. But the product hybrid extrapolates far more robustly: at its converging rate (lr 3e-4) the single-delta GatedDeltaNet extrapolates on only 1/5 seeds (L64 0.26 ± 0.19) versus gdp's 3/5. Enabling the published short-convolution (previously off) does not change this — gdp 3/3 extrapolate, gdn 1/3 — and a fair attention recipe does not rescue the transformer (§5.1).

### 5.3 Layer 3: Necessity

The product transition is distinctively necessary for the S5 state leg (§3.1), not for this composite. Last-write-wins binding is TC⁰ (a register overwrite), not the NC¹ S5 problem; the single-delta hybrid composes it at a tuned rate. We carry the product hybrid forward on measured grounds: it converges across a broader band, mostly extrapolates when tuned, and additionally supplies the state leg.

*Noise caveat.* Per-cell counts use two seeds and are noisy. At lr 1e-3 the sweep gives 2/2 where the original five-seed run gives 1/5. The claim is the band, not the per-cell rate.

We report this with its limits. The ≈45M result is a single two-point step (6M floor → 45M lift), not a scaling law or an emergence claim.

## 6. Limitations

- **Attention confound.** Every composite-scale "gdp" win is a hybrid with RoPE attention at `attention_ratio` 0.25 (two attention layers at the 45M, 8-layer depth). We run no retuned attention-free arm (`gdp_pure` collapses under the hybrid's recipe, confounding "attention is needed" with "pure-GDP is harder to train"). We credit the hybrid, not the recurrence.
- **The scale result (§5) is learnable, not robust.** The matched learning-rate sweep separates arms by convergent band, not by a categorical can/can't. Per-cell counts are seed-noisy (2 seeds/cell); the band, not the cell, is the claim. See §5 for the full sweep and the 5-seed confirmation.
- **Sₖ-state × recall is excluded.** Our composite is single-query / answer-only. Direct Sₖ trains only under dense per-token supervision (§3.1). Worked-trace supervision does not extrapolate via autoregressive generation.
- **Synthetic, small-scale, idealized.** An atomic-token mechanism instrument (≈6–45M params, single 3090). The §5 composite eval (`iso.strict_eval`) harvests the holder and value tokens from the generated span and requires both — more lenient on position than the benchmark's `score_exact`, so the scale numbers sit above what a position-strict score would give. Statistics are 2–5 seeds (std, not CIs). External validity to agentic tasks is argued, not shown.

## 7. Conclusion

**(1) The architecture lever that crosses the composite at scale is gated-delta recurrence, not attention.** Within the recurrent family, the product structure broadens the convergent learning-rate band and supplies the non-abelian state leg. At a tuned rate (lr 5e-4), the product hybrid converges robustly in-distribution (5/5) and extrapolates on a bare majority of seeds (3/5 at 4×, std 0.25; the default lr 1e-3 gave 0/5) — learnable and recipe-sensitive, not robust. The single-delta hybrid also composes this composite, but at one learning rate alone, confirming the product structure is not uniquely required for this task. The ≈45M result is a two-point step, not a scaling law.

**(2) The composite is learned as a resolve-then-recall pipeline, not two independent legs.** On holder-wrong examples the model recalls the property of the agent it actually resolved (routing ≈ 1.0 at the train length, never falling silent there), so recall is conditioned on the binding (P(value | holder wrong) ≈ 0) — and the circuit forms all-or-nothing across seeds. That conditional pipeline is what the architecture lever has to move.

The durable artifact is the instrument. It is oracle-validated and runnable. Others can rerun it and scale it to measure composition. 

## 8. References

- **Arora, S., Eyuboglu, S., Timalsina, A., Johnson, I., Poli, M., Zou, J., Rudra, A., and Ré, C.** (2023). Zoology: Measuring and Improving Recall in Efficient Language Models. ICLR 2024. arXiv:2312.04927.
- **Barrington, D. A. M.** (1989). Bounded-width polynomial-size branching programs recognize exactly those languages in NC¹. *Journal of Computer and System Sciences* 38(1):150–164.
- **Fu, D. Y., Dao, T., Saab, K. K., Thomas, A. W., Rudra, A., and Ré, C.** (2022). Hungry Hungry Hippos: Towards Language Modeling with State Space Models. ICLR 2023. arXiv:2212.14052.
- **Geva, M., Schuster, R., Berant, J., and Levy, O.** (2021). Transformer Feed-Forward Layers Are Key-Value Memories. EMNLP 2021. arXiv:2012.14913.
- **Grazzi, R., Siems, J., Zela, A., Franke, J. K. H., Hutter, F., and Pontil, M.** (2024). Unlocking State-Tracking in Linear RNNs Through Negative Eigenvalues. ICLR 2025. arXiv:2411.12537.
- **Lieber, O., Lenz, B., Bata, H., et al.** (2024). Jamba: A Hybrid Transformer-Mamba Language Model. arXiv:2403.19887.
- **Liu, B., Ash, J. T., Goel, S., Krishnamurthy, A., and Zhang, C.** (2023). Transformers Learn Shortcuts to Automata. ICLR 2023. arXiv:2210.10749.
- **Meng, K., Bau, D., Andonian, A., and Belinkov, Y.** (2022). Locating and Editing Factual Associations in GPT. NeurIPS 2022. arXiv:2202.05262.
- **Merrill, W. and Sabharwal, A.** (2023). The Parallelism Tradeoff: Limitations of Log-Precision Transformers. *TACL* 11:531–545. arXiv:2207.00729.
- **Merrill, W., Petty, J., and Sabharwal, A.** (2024). The Illusion of State in State-Space Models. ICML 2024. arXiv:2404.08819.
- **Okpekpe, D. and Orvieto, A.** (2025). Revisiting Associative Recall in Modern Recurrent Models. arXiv:2508.19029.
- **Siems, J., Carstensen, T., Zela, A., Hutter, F., Pontil, M., and Grazzi, R.** (2025). DeltaProduct: Improving State-Tracking in Linear RNNs via Householder Products. arXiv:2502.10297.
- **Wang, D., Zhu, R.-J., Abreu, S., et al.** (2025). A Systematic Analysis of Hybrid Linear Attention. arXiv:2507.06457.
- **Yang, S., Kautz, J., and Hatamizadeh, A.** (2024). Gated Delta Networks: Improving Mamba2 with Delta Rule. ICLR 2025. arXiv:2412.06464.
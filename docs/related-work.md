# Related Work

## 1. State Tracking and Expressivity

The S5 word problem — recognizing whether a sequence of generators produces the identity element in the symmetric group S5 — has become the canonical probe for non-abelian state tracking in sequence models. Liu et al. (2023) showed that Transformers can learn shortcut solutions that simulate finite-state automata with sub-linear depth, but these shortcuts do not reflect the sequential complexity of the underlying computation. The theoretical ceiling is established by circuit complexity: Merrill and Sabharwal (2023) showed that log-precision Transformers lie within TC⁰, a class closed under parallel prefix operations but unable to express NC¹-complete problems under standard conjectures; Merrill, Petty, and Sabharwal (2024) extended this characterisation to state-space models (SSMs), proving that SSMs are similarly upper-bounded by TC⁰ despite their recurrent appearance, an observation they call the "illusion of state." The NC¹-completeness of S5 is classical: Barrington (1989) proved that width-5 branching programs, whose state transitions come from S5, suffice to recognise any language in NC¹, making S5-recognition a natural hardness witness for the class. On the constructive side, Grazzi et al. (2024) proved that linear RNNs with state-transition matrices restricted to positive eigenvalues cannot solve even parity, and demonstrated empirically that extending eigenvalue range to negative values unlocks parity and broader state tracking. Siems et al. (2025) built on this by introducing DeltaProduct, which replaces the single Householder reflection of DeltaNet with a product of n_h Householder reflections per token; they show that n_h ≥ 4 is sufficient for S5-extrapolation, turning the architecture into a non-commutative prefix-product recurrence that sits in NC¹. FactWorld re-derives these isolation results under compute-matched controls and uses them as verified baselines for the composition experiments.

## 2. Associative Recall

Arora et al. (2023) introduced Zoology and the multi-query associative recall (MQAR) benchmark, quantifying that 82% of the performance gap between gated-convolution models and attention can be attributed to in-context recall, and showing that hybrid sparse-attention architectures recover most of this gap. Arora et al. (2024a) introduced BASED, which pairs a feature-map linear attention layer (for recall efficiency) with a local sliding-window softmax layer (for exact recall), demonstrating that the recall–throughput tradeoff can be balanced with a simple two-component hybrid. Arora et al. (2024b) proposed the Just-Read-Twice (JRT) architecture, a prefix-LM encoder–decoder that processes prompts non-causally and then decodes causally, closing the recall gap to within a few percent of Transformer quality while remaining linear-time; the key insight is that two passes over the context suffice for near-perfect recall on long-range retrieval benchmarks. Collectively, this line of work establishes that pure recurrences are recall-limited while a single well-placed attention layer reliably rescues recall — the same pattern FactWorld confirms on its isolated recall task before asking what happens under composition.

## 3. Linear-Attention, Delta-Rule, and SSM Families

The sub-quadratic sequence modelling landscape encompasses several families relevant to FactWorld's architecture comparisons. Gu and Dao (2023) introduced Mamba, a selective SSM with input-dependent state transitions that achieves linear-time inference; Dao and Gu (2024) unified SSMs and attention under a Structured State Space Duality framework, yielding Mamba-2 with a 2–8× training speedup. Yang et al. (2024a) developed Gated Linear Attention (GLA), which adds a data-dependent forgetting gate to linear attention while remaining hardware-efficient. Yang et al. (2024b) derived DeltaNet, a linear transformer that applies the delta rule (online gradient descent on an associative recall loss) as a per-token state update, enabling parallel training over sequence length. Yang, Kautz, and Hatamizadeh (2024) combined gating and the delta rule in Gated DeltaNet (GDN), which consistently outperforms both Mamba-2 and DeltaNet on language modelling and long-context understanding (ICLR 2025). Siems et al. (2025) then showed that chaining n_h ≥ 4 Householder reflections inside the delta-rule recurrence yields the GatedDeltaProduct (GDP) variant used as the primary architecture in FactWorld's state-tracking experiments. Beck et al. (2024) introduced xLSTM, extending LSTM with exponential gating and a matrix memory (mLSTM), providing another competitive linear-time baseline. Peng et al. (2025) presented RWKV-7 "Goose," a linear-complexity architecture with expressive dynamic state evolution, further widening the field of sub-quadratic alternatives.

## 4. Recall Hybrids: Recurrent Backbone + Soft Attention

The practical recipe for recovering recall in sub-quadratic models — interleaving a small number of full-softmax attention layers into a primarily recurrent stack — appears across both research and production systems. Fu et al. (2022) introduced H3, the first architecture to pair an SSM layer (for token recall) with a shift-SSM layer (for comparisons), establishing that two complementary SSMs can approximate many attention patterns and motivating the hybrid paradigm. Lieber et al. (2024) applied this at production scale with Jamba, a 52B Mamba–Transformer mixture-of-experts model whose layerwise Mamba/attention interleaving is explicitly designed to balance recurrent efficiency with attention-quality recall. Wang et al. (2025) provided the first systematic empirical analysis of hybridisation ratios across six linear-attention variants and two model scales (340M and 1.3B parameters over up to 100B tokens), finding that the optimal attention fraction depends on the backbone and that stronger standalone linear models do not necessarily produce stronger hybrids — a finding that informs FactWorld's choice of `attention_ratio=0.25`. At the production frontier, Qwen3-Next (Qwen Team, 2025) deploys a 3:1 GatedDeltaNet–GQA hybrid at 80B/3B-active scale, while Kimi Linear (Kimi Team, 2025) builds a 48B/3B-active hybrid around Kimi Delta Attention (KDA), an extension of GatedDeltaNet with finer-grained gating, achieving up to 6× decoding throughput and 75% KV-cache reduction at 1M context — demonstrating that the GDP-class hybrids studied here are already entering deployment.

## 5. Optimisation vs. Capacity in Recall

A potential confound in recall benchmarks is that apparent architectural limitations are actually optimisation artefacts. Okpekpe and Orvieto (2025) showed that learning rate sensitivity is severe for modern recurrent models on associative recall in a way it is not for Transformers: with default learning rates, many SSMs appear to fail at recall tasks that they can solve when properly tuned, and the paper reveals opposing scaling laws — recurrent models benefit most from increased width, while Transformers benefit more from depth. FactWorld takes this finding seriously. For the §5 composite, the transformer floor is checked against a five-point learning-rate sweep (`scripts/transformer_lr_sweep.py`) — consistent with their report that Transformers are comparatively learning-rate-robust — and a positive-control methodology (overfitting a small fixed example set before any evaluation) checks that floor performance reflects genuine difficulty rather than underfitting. Because their result is that *recurrent* recall failures are the ones most often mistaken for incapacity, the matching learning-rate sweep over the pure-recurrent recall nulls is the check their work most directly motivates, and is in progress.

## 6. Knowledge in Weights vs. Context, and Synthetic Knowledge-Base Probes

A distinct but adjacent literature asks where parametric knowledge lives in a trained model and how it can be located or edited. Geva et al. (2021) showed that Transformer feed-forward layers function as key-value memories whose keys correlate with textual patterns and whose values promote specific output distributions, providing an interpretability lens on how facts are encoded in weights. Meng et al. (2022a) introduced ROME, a rank-one model editing method that locates the specific MLP sublayer storing a factual association in GPT via causal tracing and then surgically rewrites it; Meng et al. (2022b) extended this to MEMIT, which generalises ROME to batch editing of thousands of associations across multiple layers. This line of work frames knowledge as a localised, editable quantity inside weights — precisely the axis that FactWorld's parametric-recall condition probes, where the agent→value map is fixed across training (so the network can encode it in weights) but must still be composed with a state-resolved pointer at inference time. FactWorld differs from the knowledge-editing paradigm in that it is not concerned with editing but with *retrieval under composition*: whether a model that has memorised a parametric map can access it correctly once a state-tracking step stands between the query and the lookup.

---

## How This Paper Differs

FactWorld's central contribution is orthogonal to each cluster above. The state-tracking literature (cluster 1) and the recall literature (clusters 2–3) each study one capability in isolation; they confirm that individual solutions exist but cannot see whether those solutions compose. The hybrid-architecture literature (cluster 4) shows that interleaving soft attention rescues recall, but does not ask whether the same architecture can simultaneously track non-abelian state and recall, let alone whether the two operations interfere when composed in a single query past the training length. The optimisation literature (cluster 5) addresses learning-rate confounds in single-capability probes; FactWorld applies the same rigour to composed tasks, and the memorisation diagnostic (fixed vs. random map) plays an analogous role. The knowledge-editing literature (cluster 6) asks how to locate or modify parametric knowledge; FactWorld instead asks whether a model can *use* parametric knowledge correctly when a state-tracking chain is interposed — the parametric-vs-in-context distinction is a controlled axis in the instrument rather than a side effect. The composition deficit — that state-tracking and recall are each solvable and length-extrapolable in isolation but not when composed in one query, except when recall degenerates to a memorised lookup — is a finding that no single-capability probe or hybrid-design study could have exposed, and FactWorld's oracle-validated, validity-gated instrument is what makes this claim trustworthy.

---

## References

**Arora, S., Eyuboglu, S., Timalsina, A., Johnson, I., Poli, M., Zou, J., Rudra, A., and Ré, C.** (2023). Zoology: Measuring and Improving Recall in Efficient Language Models. *arXiv:2312.04927*. Published at ICLR 2024.

**Arora, S., Eyuboglu, S., Zhang, M., Timalsina, A., Alberti, S., Zinsley, D., Zou, J., Rudra, A., and Ré, C.** (2024a). Simple linear attention language models balance the recall-throughput tradeoff. *arXiv:2402.18668*. Published at ICML 2024.

**Arora, S., Timalsina, A., Singhal, A., Spector, B., Eyuboglu, S., Zhao, X., Rao, A., Rudra, A., and Ré, C.** (2024b). Just read twice: closing the recall gap for recurrent language models. *arXiv:2407.05483*.

**Barrington, D.A.M.** (1989). Bounded-width polynomial-size branching programs recognize exactly those languages in NC¹. *Journal of Computer and System Sciences*, 38(1):150–164.

**Beck, M., Pöppel, K., Spanring, M., Auer, A., Prudnikova, O., Kopp, M., Klambauer, G., Brandstetter, J., and Hochreiter, S.** (2024). xLSTM: Extended Long Short-Term Memory. *arXiv:2405.04517*.

**Dao, T. and Gu, A.** (2024). Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality. *arXiv:2405.21060*. Published at ICML 2024.

**Fu, D.Y., Dao, T., Saab, K.K., Thomas, A.W., Rudra, A., and Ré, C.** (2022). Hungry Hungry Hippos: Towards Language Modeling with State Space Models. *arXiv:2212.14052*. Published at ICLR 2023.

**Geva, M., Schuster, R., Berant, J., and Levy, O.** (2021). Transformer Feed-Forward Layers Are Key-Value Memories. *arXiv:2012.14913*. Published at EMNLP 2021.

**Grazzi, R., Siems, J., Zela, A., Franke, J.K.H., Hutter, F., and Pontil, M.** (2024). Unlocking State-Tracking in Linear RNNs Through Negative Eigenvalues. *arXiv:2411.12537*. Published at ICLR 2025.

**Gu, A. and Dao, T.** (2023). Mamba: Linear-Time Sequence Modeling with Selective State Spaces. *arXiv:2312.00752*.

**Kimi Team.** (2025). Kimi Linear: An Expressive, Efficient Attention Architecture. *arXiv:2510.26692*.

**Lieber, O., Lenz, B., Bata, H., Cohen, G., Osin, J., Dalmedigos, I., Safahi, E., Meirom, S., Belinkov, Y., Shalev-Shwartz, S., et al.** (2024). Jamba: A Hybrid Transformer-Mamba Language Model. *arXiv:2403.19887*.

**Liu, B., Ash, J.T., Goel, S., Krishnamurthy, A., and Zhang, C.** (2023). Transformers Learn Shortcuts to Automata. *arXiv:2210.10749*. Published at ICLR 2023.

**Meng, K., Bau, D., Andonian, A., and Belinkov, Y.** (2022a). Locating and Editing Factual Associations in GPT. *arXiv:2202.05262*. Published at NeurIPS 2022.

**Meng, K., Sharma, A.S., Andonian, A., Belinkov, Y., and Bau, D.** (2022b). Mass-Editing Memory in a Transformer. *arXiv:2210.07229*. Published at ICLR 2023.

**Merrill, W. and Sabharwal, A.** (2023). The Parallelism Tradeoff: Limitations of Log-Precision Transformers. *Transactions of the Association for Computational Linguistics*, 11:531–545. arXiv:2207.00729.

**Merrill, W., Petty, J., and Sabharwal, A.** (2024). The Illusion of State in State-Space Models. *arXiv:2404.08819*. Published at ICML 2024.

**Okpekpe, D. and Orvieto, A.** (2025). Revisiting associative recall in modern recurrent models. *arXiv:2508.19029*.

**Peng, B., Zhang, R., Goldstein, D., Alcaide, E., Du, X., Hou, H., Lin, J., Liu, J., Lu, J., Merrill, W., et al.** (2025). RWKV-7 "Goose" with Expressive Dynamic State Evolution. *arXiv:2503.14456*.

**Qwen Team.** (2025). Qwen3 Technical Report. *arXiv:2505.09388*. [Note: Qwen3-Next is a subsequent hybrid-attention release from the same team; no separate arXiv paper was located at time of writing — cite as Qwen Team 2025 / Qwen3-Next model release.]

**Siems, J., Carstensen, T., Zela, A., Hutter, F., Pontil, M., and Grazzi, R.** (2025). DeltaProduct: Improving State-Tracking in Linear RNNs via Householder Products. *arXiv:2502.10297*.

**Wang, D., Zhu, R.-J., Abreu, S., Shan, Y., Kergan, T., Pan, Y., Chou, Y., Li, Z., Zhang, G., Huang, W., and Eshraghian, J.** (2025). A Systematic Analysis of Hybrid Linear Attention. *arXiv:2507.06457*.

**Yang, S., Wang, B., Shen, Y., Panda, R., and Kim, Y.** (2024a). Gated Linear Attention Transformers with Hardware-Efficient Training. *arXiv:2312.06635*. Published at ICML 2024.

**Yang, S., Wang, B., Zhang, Y., Shen, Y., and Kim, Y.** (2024b). Parallelizing Linear Transformers with the Delta Rule over Sequence Length. *arXiv:2406.06484*.

**Yang, S., Kautz, J., and Hatamizadeh, A.** (2024c). Gated Delta Networks: Improving Mamba2 with Delta Rule. *arXiv:2412.06464*. Published at ICLR 2025.

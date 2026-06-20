# Follow-on findings: where does the parametric composite actually break?

Post-publication, segregated investigation (not in the shipped paper). Motivating question: it seemed
surprising that the model "couldn't look up a parametric fact and update it with state-tracking + in-context
knowledge." This localizes the break and **corrects the framing**.

## TL;DR

**The parametric dereference is not the wall. The non-abelian state leg is.** Decomposed, recalling an
agent's fact from the *weights* is exactly as reliable as copying it from the *prompt*, given a resolved
pointer (`P(value | holder✓) = 1.00`, `route = 1.00` in both arms). The composition gap is entirely a
**state-tracking-under-single-query-supervision** problem; parametric vs in-context recall is irrelevant to it.
Confirmed constructively: add dense per-step state supervision and the non-abelian composite is **solved**
end-to-end (e2e 1.00 at train length), with the **parametric** arm extrapolating at least as well as
in-context (0.95±0.06 vs 0.55±0.37 at 4×). **State supervision is the lever; parametric recall is free.**
But the lever is steep: a supervision-sparsity sweep shows a **cliff** — the circuit forms only with a state
checkpoint at ~every other step (K≤2), collapses to chance by K=4, and needs *fully* dense supervision to
extrapolate in horizon. Weaning (curriculum/mixed) can **internalise** the circuit (run with no scratchpad) but
only at the trained length — a clean dissociation: externalised/scratchpad tracking extrapolates in horizon but
keeps the crutch; internalised tracking works at trained length but is brittle to horizon (length-gen **or**
no-scratchpad, not both). A scale check shows the horizon wall is **flat at floor up to 44.8M** (nearly 8×, even
LR-tuned); at a 70M control point a **weak, LR-sensitive lift** appears (0.41 at lr 1e-3, floor at 5e-4), so the
wall is largely a **learnability/structural** limit rather than a capacity one within reach — though capacity is
not irrelevant at the ceiling. Net: the composition gap is a state-supervision problem throughout, and the open
frontier is *length-robust internalised* state-tracking (pure scale is not the lever; horizon-extension
curriculum or a compressed recurrent state-summary are the live candidates). This sits in the *learnability*
dimension that Mozer, Siddiqui & Liu (arXiv 2604.17121) explicitly bracket out of their expressivity-focused
state-tracking thesis — see Related Work.

## Evidence

### Ladder (`ladder.py`, gdp_hybrid, 3 seeds) — parametric recall, fixed map, facts omitted
| rung | adds | L16 | L64 |
|---|---|---|---|
| R0 | literal key, no state | **1.00** | — |
| R1 | abelian binding, NO CoT | **0.83** | 0.57 |
| R2 | abelian binding, CoT | 0.82 | 0.57 |
| R3a | non-abelian state, NO CoT | 0.19 (floor) | 0.21 |
| R3b | non-abelian state, CoT | 0.21 (floor) | 0.21 |

- A *computed* (abelian-resolved) pointer dereferences weight memory fine (R1=0.83, latent, no CoT). So
  "parametric recall can't be keyed by anything the model computes" is **false**.
- The wall is specific to **non-abelian state + recall**, and CoT does not move it.

### Decomposition (`decompose_r3.py`, 5 seeds, CoT) — in-context vs parametric, identical chains/map
| arm | holder | value | P(v\|h✓) | route(h✗) |
|---|---|---|---|---|
| in-context (facts rendered) | 0.21 | 0.21 | 1.00 | 1.00 |
| parametric (facts omitted)  | 0.21 | 0.21 | 1.00 | 1.00 |

- Identical to the digit. `holder == value == both`, `P(v|h✓)=1.0`, `route=1.0` everywhere: the **recall
  circuit is perfectly formed** (name an agent → emit *that* agent's value, parametric or in-context); only
  the **holder** (non-abelian state) sits at chance (1/k = 0.20).
- The in-context arm is the positive control: it floors too, matching the known result that non-abelian
  state-tracking needs **dense per-token supervision** (`scripts/dense_s5.py`) and floors under answer-only
  single-query supervision — regardless of where the fact lives.

## Reconciliation with the shipped paper

§4's line "with parametric (in-weights) facts gdp collapses … like every other arch" reads as *parametric
recall* being the obstacle. The decomposition shows the operative variable is the **state leg**, not the
recall leg. (Tension with the paper's parametric line — R1 abelian+parametric = 0.83 does not "collapse" —
is RESOLVED on the scoring axis: `recon_b1.py` confirms position-strict == value-scan to the digit (L16
0.83/0.83, L64 0.57/0.57), since R1's answer is a single token at the answer position. So the difference is
NOT a metric artifact; it is a configuration difference — the paper's flagship composite couples parametric
recall with a harder/longer binding. We report the abelian dereference on its own terms and make no public
correction to the companion composite.)

## Capstone — dense supervision rescues the composite, parametric included (`dense_capstone.py`, 5 seeds)

Constructive confirmation: interleave **dense per-step holder supervision** (oracle `hard_trace`) into the
non-abelian composite — the affirmative version of the dense_s5 lesson. Contrast baseline: R3b (final-only
CoT) floored at 0.20 for both arms.

| arm | L | dense_h | val\|trace | e2e_holder | e2e_value |
|---|---|---|---|---|---|
| in-context | 16 | 1.00 | 1.00 | 1.00 | 1.00 |
| in-context | 64 | 0.96 | 1.00 | 0.55±0.37 | 0.55±0.37 |
| parametric | 16 | 1.00 | 1.00 | 1.00 | 1.00 |
| parametric | 64 | 1.00 | 1.00 | **0.95±0.06** | **0.95±0.06** |

- **State leg lifts off the floor:** `dense_h` 0.20 (R3b) → **1.00** at train length, ~0.96–1.00 at 4×.
- **Composite solved end-to-end at train length** (`e2e = 1.00`) for **both** in-context and parametric.
- **Parametric is not the bottleneck — it extrapolates *better*:** at 4× free-running, parametric e2e is
  0.95±0.06 (tight) vs in-context 0.55±0.37 (bimodal — some seeds collapse decoding a 64-long interleaved
  trace). A fixed weight lookup adds no per-step ambiguity; in-context copy over a long trace has more to
  go wrong. `e2e_holder == e2e_value` throughout → recall stays gated on (and free given) the state.

**Conclusion (confirmed both ways):** the composition gap is a **state-supervision / credit-assignment**
problem, not a parametric-binding problem. Supervise the intermediate state densely and the non-abelian
composite is solved — and the parametric variant rides along at least as well as in-context. State
supervision is the lever; parametric recall is free.

## How dense must the supervision be? — the agentic catch (`supervision_sweep.py`, 5 seeds)

The capstone uses a per-step oracle state trace — privileged info an agent won't have. So the transferable
question is the curve between answer-only (floors) and dense (solves): supervise the holder every **K** events
(+ always the final), parametric recall, gdp_hybrid.

| K | checkpoints / 16-ep | L16 e2e | L64 e2e | converged |
|---|---|---|---|---|
| 1 (dense) | 16 | **1.00** | 0.78±0.24 | 5/5, 4/5 |
| 2 | 8 (every other) | **0.98±0.01** | 0.29±0.20 | 5/5, 1/5 |
| 4 | 4 (every 4th) | **0.19** (floor) | 0.19 | 0/5 |
| 8 | 2 | 0.22 | 0.21 | 0/5 |
| inf (answer-only) | 1 | 0.19 | 0.20 | 0/5 |

**It is a cliff, not a slope.** The circuit forms in-distribution only down to a checkpoint **every other
step** (K≤2); at K=4 it collapses to chance — and at K≥4 even `dense_h` (accuracy *at the labelled slots*) is
at floor, so the recurrence fails to discover the permutation computation *at all*, all 5 seeds. Length
**extrapolation** is stricter still: it needs **fully dense** supervision (K=1); K=2 solves the train length
but falls to 0.29 at 4×.

**Agentic reading (the honest, sobering one).** There is no comfortable middle where "a few checkpoints"
suffice. Forming compositional latent state-tracking here requires **near-dense process supervision** (a state
oracle at ~every step) and **fully dense** to generalise in horizon. Sparse outcome/answer supervision — the
regime an agent actually operates in — gives *zero* traction (chance, every seed). So the capstone's success
sits at the K=1 extreme of a cliff whose floor begins at K=4; the dense-supervision recipe does **not**
transfer to the sparse-reward agentic setting. The open, genuinely-transferable problem is whether a
**curriculum** (dense supervision to *form* the circuit, then wean to sparse) can move that cliff — i.e.
whether the circuit, once formed, can be run **without** the scratchpad (internalised). Tested next.

## Can weaning internalise the circuit? — curriculum (`curriculum.py`, 6000 steps, 5 seeds)

Gradual scratchpad removal. `anneal` = supervision density K scheduled 1→2→4→8→inf over training (dense forms,
sparse internalises); `mixed` = random K per example (order-agnostic control). Agentic target = **answer-only**
eval (no scratchpad). Baselines: from-scratch answer-only = **0.20 floor (0/5)**; dense-scratchpad K=1 = 1.00 /
0.78 (but keeps the crutch).

| arm | answer-only L16 | answer-only L64 | dense (scratchpad) L16 |
|---|---|---|---|
| anneal (curriculum) | 0.43±0.26 (1/5) | 0.21 (0/5) | 0.85 (4/5) |
| mixed (no order) | 0.66±0.28 (3/5) | 0.22 (0/5) | 1.00 (5/5) |

Three findings, in increasing importance:
1. **Internalisation is possible but seed-fragile.** Some seeds answer with *no* scratchpad and decisively beat
   the floor (mixed s1/s4 = 1.00 at L16) — so weaning is a real mechanism — but only 3/5 (mixed) / 1/5 (anneal).
2. **Order is not the lever.** The dense→sparse *curriculum* gives no advantage; `mixed` ≥ `anneal` on every
   metric. What helps is *exposure to both densities* (form the circuit AND practise without the crutch), not
   the schedule. The "compose short sub-sessions" intuition survives only in the weak form "train on a mix of
   dense-short and sparse examples," not as an ordered curriculum.
3. **Internalisation does not extrapolate in horizon — the key wall.** Answer-only L64 = floor for *both* arms
   (0/5), even on seeds that internalise perfectly at L16. This is a clean dissociation:
   - **externalised** (scratchpad) state-tracking *extrapolates* in length (K=1: L64 0.78) but keeps the crutch;
   - **internalised** (no scratchpad) is achievable at the *trained* length (L16) but is *brittle to horizon*.
   At this scale you can have length-generalisation **or** no-scratchpad, **not both**.

**Agentic verdict.** Ian's "a long session is a composition of short, well-supervised sub-sessions" gets a
**qualified yes at fixed horizon**: dense-then-sparse exposure can internalise the per-step computation so the
model answers without a scratchpad (beating the from-scratch floor). But the payoff that would make it useful
for *arbitrarily long* sessions — running the internalised tracker past the trained horizon — does **not** come
for free; the no-scratchpad circuit is the one that fails to extrapolate. So composing short sessions buys you
internalisation, not unbounded length. Closing that last gap (length-robust *internalised* state-tracking) is
the real open problem — candidates: larger scale, explicit horizon-extension curriculum, or a hybrid that keeps
a *compressed* recurrent state-summary rather than a full token scratchpad.

## Is the horizon wall a capacity limit? — scale check (`scale.py`, mixed recipe, 3 seeds)

Scale the mixed-density recipe across 5.7M / 18.5M / 44.8M (the last = the paper's ~45M).

| scale | answer-only L16 (internalise) | **answer-only L64 (the wall)** | dense L64 (externalised) |
|---|---|---|---|
| 5.7M  | 0.67±0.24 (2/3) | **0.23±0.04 (0/3)** | 0.45±0.38 (1/3) |
| 18.5M | 0.99±0.01 (3/3) | **0.20±0.02 (0/3)** | 0.71±0.36 (2/3) |
| 44.8M | 0.95±0.07 (3/3) | **0.22±0.05 (0/3)** | 0.68±0.39 (2/3) |

Scale has **three dissociable effects**: (a) internalisation *at the trained length* goes from fragile to
reliable (L16 0.67 → 0.99/0.95, 2/3 → 3/3); (b) externalised (scratchpad) extrapolation improves then plateaus
(dense L64 0.45 → ~0.70); (c) **the internalised horizon wall does not move at all** — answer-only L64 is flat
at floor across an 8× parameter increase (0.23 / 0.20 / 0.22, 0/3 everywhere).

**Verdict (refined below): up to 44.8M on this fixed recipe the horizon wall is NOT a capacity limit.**
Internalised state-tracking past the trained horizon is a **learnability/structural** failure here, not a
model-size one — more parameters buy reliability *at length seen*, nothing *beyond* it. So "just scale" is not
the lever at these scales; the live fixes are horizon-extension curriculum and the compressed recurrent
state-summary. (The LR-tuned + 70M control below refines this: a *weak* lift appears at 70M, so capacity is not
strictly irrelevant — but it is far from solving and not the lever.)

## Moving the wall — horizon curriculum works, token re-anchoring doesn't (`horizon.py`, `coarse.py`, 3 seeds, 18.5M)

**Horizon-extension curriculum (`horizon.py`) — WORKS.** Grow training lengths 8→16→32→48→64 (mixed density),
then eval internalised answer-only:

| eval | answer-only e2e | note |
|---|---|---|
| L16 | 1.00 (3/3) | trained edge |
| L64 | **0.94 (3/3)** | in extended range — was 0.20 floor when trained ≤16 |
| L128 | 0.48 (2/3) | 2× max train (truly OOD) |

The internalised horizon wall — flat at floor across 8× *scale* — is moved decisively by training the
recurrence at length: floor → 0.94 at 4×, with partial extrapolation (0.48) *beyond* the trained horizon.
The limit was never capacity; it was that the recurrence had never been trained to run that many steps.

**Compressed token-level full-state re-anchoring (`coarse.py`) — FAILS.** Emit a bounded k-token digest of the
full role→agent state every C events and re-anchor; free-running composite eval:

| condition | L64 | L128 |
|---|---|---|
| full_C8 | 0.15 (0/3) | 0.22 (0/3) |
| full_C16 | 0.20 (0/3) | 0.19 (0/3) |
| single_C8 (control) | 0.23 (0/3) | 0.21 (0/3) |

Floor everywhere; full-state ≈ single-role control. **Why:** generating the digest is *itself* the hard
non-abelian state step, and free-running the model feeds its own (sometimes wrong) digests forward, so error
compounds across summaries exactly as across unsupervised steps. Externalising state into tokens re-imports the
drift it was meant to bound.

**Lesson:** close the wall by extending the *latent recurrence's* trained horizon, not by adding explicit
token-level state summaries. "Compose short sub-sessions" works when each sub-session trains the recurrence to
run longer; it fails when it asks the model to write and re-read its own state.

## Related work — positioning against Mozer, Siddiqui & Liu, *The Topological Trouble With Transformers* (arXiv 2604.17121)

A 2026 position paper (no experiments) that is the near-perfect framing anchor for this follow-on, and that our
results **build on and sharpen**. Its thesis: feedforward transformers have a *topological* limit on state
tracking — state is pushed deeper into the layer stack each step until depth is exhausted — so the field should
move from **externalised thought traces (CoT/scratchpad) to implicit recurrent activation dynamics**. It offers
a taxonomy (recurrence axis × input-tokens-per-recurrence-step) and names enhanced SSMs and *coarse-grained
recurrence* as directions. It places **DeltaProduct (Siems et al. 2025)** and negative-eigenvalue DeltaNet
(Grazzi et al.) in its most-expressive cells — i.e. **exactly our `gdp_hybrid` architecture**.

How our follow-on relates:
- **We fill the dimension it brackets out.** It is about *expressivity / constructability* and states plainly
  (re Merrill & Sabharwal) that the log-depth proof *"addresses only the constructability of solutions, not
  their learnability."* Our entire arc — the supervision cliff, internalisation, the horizon dissociation, and
  now the scale check — is the **learnability** map it declares open. Complementary, not competing.
- **We empirically push back on a specific claim.** It argues (Table 1 discussion, p.8) that coupling
  step-recurrence with attention *"prevents credit assignment bottlenecks that arise when training traditional
  RNNs (Ke et al. 2018)."* Our sweep shows a credit-assignment **cliff persists** in precisely that coupled
  `gdp_hybrid` (K=4 → chance, all seeds). The coupling does not, on its own, dissolve the credit-assignment
  problem; supervision density does.
- **Our failure is not their mechanism.** Depth-exhaustion is their diagnosis for *feedforward* models; our
  hybrid is the *recurrent* class they advocate. The scale check shows the horizon wall flat to 44.8M (only a
  weak lift at 70M) — so it is largely *not* an expressivity/depth limit (which scale would relax), but the
  learnability limit they set aside. This is the cleanest statement of where our contribution sits relative to theirs.
- **Convergent fix.** Their "coarse-grained recurrence" direction is the same compressed recurrent state-summary
  we independently propose for the horizon wall — they motivate it, we provide the testbed.

Cite as the expressivity/architectural framing our learnability study sits against (and cite the p.8
credit-assignment claim as the specific point our cliff contradicts). Not as empirical support — it has none.

## RL vs static — outcome reward does NOT climb the cliff (`rl_grpo.py`, `rl_flicker.py`)

Is "process supervision required" an SFT artifact? Agentic training is RL. Test: answer-only SFT warmup (at the
floor) → GRPO, 0/1 reward on the composite answer, free scratchpad, d256.
- `rl_grpo.py` (n=3, 1500 GRPO steps): RL value = 0.19±0.04 ≈ SFT floor; holder 0.05±0.07. 2/3 seeds flat, 1
  seed marginal flicker (holder 0.15-0.23) that did not yield correct values.
- `rl_flicker.py` (powered-up, 7500 steps, finer logging): seed-0 reward DEAD FLAT at chance (0.195-0.201)
  across 5000 steps; mid-evals v 0.16→0.21→0.21, holder 0.00 throughout — no climb. (Stopped after the seed-0
  trajectory settled the climb-vs-noise question; full n=5×7500 re-runnable if publication needs it.)

**Verdict:** outcome-reward RL does not climb the cliff. Mechanism = exploration barrier: reward variance never
exceeds chance, so GRPO advantages are noise and there is no gradient to bootstrap. So "process supervision
required" is robust across paradigms (static SFT AND outcome RL both fail without it). Caveat: one arch, one RL
recipe; reward shaping / curriculum untested.

## Construct-validity bridge — the cliff appears in CODE-EXECUTION clothing (`code_trace.py`, 3 seeds, 18.5M)

Does non-abelian state matter beyond the abstract S5 word problem? Re-render the IDENTICAL dynamics as a
variable-swap execution trace in CWM surface grammar (vars hold values, swap/cycle ops, full-state snapshot
every K ops, query a variable's final value). Same density sweep as `supervision_sweep.py`.

| K (snapshot/op) | L16 | L64 |
|---|---|---|
| 1 (every op) | 1.00 | **0.97** |
| 2 | 0.99 | **0.89** |
| 4 | 0.87 | **0.50** |
| 8 | 0.72 | **0.29** |
| inf (answer-only) | 0.46 (bimodal) | **0.25** |

**The density cliff reproduces in code-execution clothing**, cleanest in length-extrapolation (L64: 0.97→0.25
as snapshots thin). Two honest nuances: (1) the in-distribution (L16) threshold sits at SPARSER K than the
role-rendered single-role sweep — full-state snapshots carry more info per checkpoint, so richer process
supervision needs lower density (mechanism-consistent); (2) K=inf is bimodal — the full final-state snapshot
turns the task into "track-to-end, emit state, copy the variable," which a minority of seeds solve. Verdict:
the wall is non-abelian COMPOSITION, not the role-permutation rendering — it plants directly in the code regime
(tracking a variable through reassignments = the same non-shortcuttable composition, same near-dense process
supervision). Anchors to CWM (arXiv 2510.02387) as the at-scale real-code instance; see Related Work.

## Supervision density × training horizon are ORTHOGONAL (`sup_horizon.py`, 3 seeds, 18.5M)

Does training the recurrence at length lower the supervision density it needs? Run the density sweep K at
LONG-horizon training (lengths ≤64), compare the cliff threshold to the ≤16 baseline (`supervision_sweep.md`).

| K | L64 (long-horizon train) | L128 | ≤16-baseline L64 |
|---|---|---|---|
| 1 | 0.98 | 0.97 | 1.00 |
| 2 | 0.99 | 0.98 | **0.29** |
| 4 | 0.19 (0/3) | 0.22 | 0.19 |
| 8 | 0.18 (0/3) | 0.18 | 0.22 |
| inf | floor | floor | 0.20 |

**The density threshold is horizon-INDEPENDENT.** K≥4 floors regardless of training length (cliff stays between
K=2 and K=4, same as the ≤16 baseline) — so training at length does NOT let sparser supervision form the
circuit. What long-horizon training *does* buy is **extrapolation at the densities that already form**: K=2 goes
from ≤16-baseline 0.29 → 0.98 at L64 (and 0.98 at L128). So **supervision density and training horizon are
orthogonal levers**: density gates whether the circuit *forms* (fixed threshold ~K=2); horizon gates how far it
*extrapolates* once formed. This unifies the two findings — density = the §4 cliff, horizon = the §6 curriculum
— under one frame, and they don't trade off.

## Scale-done-right — LR-tuned + 70M control point (`scale_tuned.py`, 2 seeds, 8000 steps)

The scale check (scale.py) held lr=1e-3 fixed and topped at 44.8M — so "flat at floor" could be under-training.
Re-run the largest scales LR-tuned ({1e-3,5e-4}) + a 70M control point + more steps. Headline = answer-only L64.

| scale | lr=1e-3 | lr=5e-4 |
|---|---|---|
| 44.8M | 0.24±0.04 | 0.19±0.05 |
| **70M** | **0.41±0.05** | 0.24±0.09 |

**Refines the capacity claim (honestly).** 44.8M floors at BOTH LRs (so under-training isn't the explanation
there). At 70M the wall shows a **weak, LR-sensitive partial lift**: 0.41 at lr=1e-3 (both seeds above floor)
but back to floor at lr=5e-4 (one seed collapsed). So it is NOT "capacity ruled out" — something marginal begins
to help around 70M (1.5× past the old ceiling). But it's far from solving (0.41 vs 1.0) and not robust across LR.
Net: capacity is not the lever (the wall is substantially intact at 70M and a learnability limit where flat),
but we cannot rule out stronger relief at much larger scale. §5/abstract/conclusion softened accordingly.

## Horizon mechanism — the internalized cap tracks max-train-length (`horizon_mech.py`, 3 seeds, 18.5M)

What sets the internalized length cap? Train internalized (mixed-density) to several Lmax, eval answer-only at
Lmax (in-range) and 2×Lmax (OOD).

| Lmax | in-range (L=Lmax) | OOD (2×Lmax) |
|---|---|---|
| 16 | 0.82±0.07 | 0.27 (L32) |
| 32 | 0.62±0.11 | 0.23 (L64) |
| 48 | 0.37±0.29 | 0.23 (L96) |
| 64 | 0.64±0.14 | 0.30 (L128) |

**The solvable cap tracks the max training length** — the internalized circuit answers within ~[0, Lmax] and
floors at ~2×Lmax (OOD 0.23–0.30 throughout) — which mechanistically confirms the §6 lesson (extend reach by
training the recurrence longer). Honest caveat: in-range accuracy is **solvable but seed-fragile and
high-variance across Lmax** (0.37–0.82, std up to 0.29, no clean monotone trend; the Lmax=48 row is a low,
bimodal outlier). So "train longer to extend the cap" holds in that the solvable region grows with Lmax and
floors at ~2×, but internalizing long non-abelian chains stays noisy/fragile at this scale — consistent with the
state circuit being length-sensitive and internalized > externalized in difficulty.

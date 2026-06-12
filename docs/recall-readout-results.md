# Recall read-out differential (1-hop vs deferred) — `scripts/recall_fair.py`

**Question.** FactWorld's recall tasks use a *deferred* read-out: the queried value is supervised at a
separated answer position, not emitted as the next token after the key (canonical 1-hop MQAR; Arora et al.,
2023). Earlier baselines showed the transformer at ~0.48 on this — is that "attention can't recall"
(contradicting Zoology), a recipe confound (n_heads, init), or a genuine *regime* difference (the deferred
read-out)? This isolates format from recipe on a clean 1-of-2 copy (pool 2; chance-given-key = 0.5,
absolute floor 1/64 = 0.016), atomic vocab, identical optimizer/steps (30k, cosine, lr 1e-3), 2 seeds.

Three read-outs for a query on key `k` with value `v`:
- `onehop` — `[… k v]`, target `v` at the key position (canonical Zoology MQAR; 1-hop).
- `defsep` — `[… k SEP]`, target `v` at the SEP token right after the key (a *fair* 2-hop deferral).
- `defpad` — `[… QUERY k ANS PAD]`, target `v` at the trailing PAD slot (the `recall_copy_v1` format).

## Result

| arch | read-out | heads | acc | seeds |
|---|---|---|---|---|
| transformer | onehop (1-hop) | 4 | **1.000** | 1.00 / 1.00 |
| transformer | onehop (1-hop) | 8 | **1.000** | 1.00 / 1.00 |
| transformer | defsep (fair 2-hop) | 8 | 0.759 | 0.52 / 1.00 |
| transformer | defpad (our format) | 8 | 0.638 | 0.75 / 0.52 |
| gdp_pure | onehop (1-hop) | 4 | **1.000** | 1.00 / 1.00 |
| **gdp_pure** | **defpad (our format)** | 4 | **1.000** | 1.00 / 1.00 |

## Reading

1. **The transformer is not recall-incapable, and n_heads is not the cause.** It solves canonical 1-hop
   MQAR *perfectly and robustly* at both 4 and 8 heads (1.00 / 1.00). This exonerates the n_heads=4 default
   and any "wiring bug," and is consistent with Zoology (attention solves MQAR) — we are **not**
   contradicting that literature.
2. **The deferral is the whole story; the PAD slot is not a separate pathology.** `defpad` (0.638) ≈
   `defsep` (0.759) — both seed-fragile, overlapping seed spreads — so our `recall_copy_v1` format is
   *representative* of the deferred-read-out regime, not a broken construct. The contrast that matters is
   1-hop (1.00, robust) vs any 2-hop (~0.5–0.76, fragile).
3. **The deferred read-out is "free" for the product recurrence and only fragilely learnable for
   attention.** On the identical `defpad` task, `gdp_pure` is robust (1.00 / 1.00) where the transformer is
   seed-fragile (0.638; forms the match-then-copy circuit on one seed, floors on the other). The recurrence
   carries the binding in state and reads it out at an arbitrary later position; attention must re-form a
   match-then-copy circuit across the answer gap and does so unreliably at this budget.

**Implication for the paper.** The §3.2 baseline numbers (transformer 0.48, gdn_hybrid 0.51) are the
*deferred-read-out* regime at heads=4, not a recall ceiling. Frame the recurrent advantage as
**read-out-deferred binding** (the FactWorld thesis: state, not recall), and lead the architectural claim
with the unconfounded attention-free comparison (gdp_pure vs gdn_pure), which does not depend on attention's
read-out at all. Report the 1-hop control so the reader sees attention *does* solve canonical MQAR.

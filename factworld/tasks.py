"""FactWorld task suite — frozen, versioned, scalable benchmark tasks (the reusable "body").

This is the layer that turns the bespoke experiment scripts into a *benchmark*: each task is a FROZEN
`TaskSpec` (pinned seed, version, explicit difficulty knobs and train/OOD-length splits), examples are
generated deterministically, and there is ONE canonical metric — **relaxed match** of the answer span
(whitespace / trailing-period invariant; see ``score_relaxed``). Exact match, semantic containment,
and last-*n* extraction are reported as diagnostics, not headline scores. Difficulty knobs (k,
n_objects, recall pool, lengths) are exposed so a task can be scaled to genuinely stress larger
models; the canonical registry pins reference instances.

Label discipline (inherited from the instrument): every example's gold answer comes from the symbolic
**oracle**, never from parsing rendered text — so labels cannot leak. This module is torch-free (data
generation needs no GPU).

  from factworld.tasks import CANONICAL, generate, score_exact
  spec = CANONICAL["composite_v1"]
  train = generate(spec, "train", n=8000)
  test  = generate(spec, "test", length=64)          # held-out OOD-length split, fixed seed
  acc   = sum(score_exact(pred, ex.answer) for pred, ex in zip(preds, test)) / len(test)
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field, replace

from .config import WorldConfig
from .oracle import Oracle
from .render import Renderer
from .world import Event, World

SUITE_VERSION = "1.0"

# The one canonical metric reported as the headline score (see ``score_relaxed``). The other scorers
# (exact / contains / last_n) are diagnostics. ``runner.evaluate_task`` reads this so the runner,
# the CLI, and the reports all agree on what "the score" means.
CANONICAL_METRIC = "relaxed"


@dataclass(frozen=True)
class TaskSpec:
    """A frozen, reproducible benchmark task. Difficulty knobs are explicit and scalable."""
    name: str
    family: str                       # 'recall' | 'binding' | 'composite' | 's5'
    version: str = SUITE_VERSION
    seed: int = 0
    # 'benchmark' = a scored, discriminating task; 'control' = a positive control / isolation task that is
    # degenerate as a capability score (e.g. memorized-map recall); 'experimental' = correct construct but
    # not reliably trainable in this harness yet (see s5_v1). Only 'benchmark' tasks are in REPORTED.
    kind: str = "benchmark"
    # world / breadth knobs
    k: int = 5                         # agents (= roles for s5); also the recall pool unless recall_pool set
    n_objects: int = 8
    value_vocab_size: int = 64
    n_objects_active: int = 4          # objects actually used in a give-stream (binding working set)
    # recall knobs
    recall_pool: int | None = None     # composite/recall: # facts presented (1-of-N); None -> use k
    memorized_recall: bool = True      # fixed agent->value map (memorizable) vs random per example (copy)
    # length / horizon knobs (the extrapolation axis)
    train_lengths: tuple = (4, 8, 16)  # binding-chain / permutation-history lengths in training
    eval_lengths: tuple = (16, 32, 64) # held-out OOD lengths
    worked_trace: bool = False         # s5/composite: emit the oracle state trajectory as a scratchpad

    def scaled(self, **knobs) -> "TaskSpec":
        """Return a harder/easier variant (e.g. spec.scaled(k=64, recall_pool=64, eval_lengths=(32,128)))."""
        return replace(self, **knobs)


@dataclass
class Example:
    prompt: str        # the model input (the full query, ending in '?')
    answer: str        # the exact expected continuation (space-separated atomic tokens)
    length: int        # the difficulty coordinate this example was drawn at
    meta: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# deterministic RNG: keyed by (spec, split, length, index) so train/test never overlap and are fixed
# ---------------------------------------------------------------------------
def _rng(spec: TaskSpec, split: str, length: int, idx: int) -> random.Random:
    return random.Random(f"factworld|task|{spec.name}|{spec.version}|{spec.seed}|{split}|{length}|{idx}")


def _world(spec: TaskSpec) -> tuple[World, Renderer, Oracle]:
    wc = WorldConfig(seed=spec.seed, n_entities=8, n_attributes=2,
                     value_vocab_size=spec.value_vocab_size, n_objects=spec.n_objects,
                     n_locations=6, k=spec.k)
    w = World(wc)
    return w, Renderer(), Oracle(w)


def build_world(spec: TaskSpec) -> tuple[World, Renderer]:
    """Public: the (World, Renderer) a task is built on — e.g. to build a tokenizer covering it."""
    w, r, _ = _world(spec)
    return w, r


def _fixed_origins(spec: TaskSpec, w: World) -> dict:
    rng = random.Random(f"factworld|origins|{spec.name}|{spec.seed}")
    return dict(zip(w.agents, rng.sample(list(w.value_vocab), spec.k)))


def _param_map(spec: TaskSpec, w: World) -> dict:
    """The 'in-weights' agent→value map for the conflict family — constant across training so a model
    memorizes it, and the in-context value at test deliberately contradicts it."""
    rng = random.Random(f"factworld|parammap|{spec.name}|{spec.seed}")
    return {a: rng.choice(list(w.value_vocab)) for a in w.agents}


def _render_answer(core: str) -> str:
    """Render the answer span with attached punctuation (e.g. `g4.`, `g0 v109.`) to match the
    natural-language prompt style."""
    return f"{core}."


# ---------------------------------------------------------------------------
# per-family example builders
# ---------------------------------------------------------------------------
def _ex_recall(spec, w, r, fixed_origins, rng, split, length, idx):
    """Associative recall (1-of-N). memorized -> fixed map (fixed pool); in-context-copy -> the pool size
    is the LENGTH axis, so `eval_lengths` grows the distractor set (a real recall-extrapolation axis)."""
    if spec.memorized_recall:
        pool = spec.recall_pool or spec.k
        chosen = list(w.agents[:pool]); origins = fixed_origins
    else:
        pool = min(length, len(w.agents))                          # 1-of-N where N = length (the difficulty axis)
        chosen = rng.sample(list(w.agents), pool)
        origins = dict(zip(chosen, rng.sample(list(w.value_vocab), pool)))
    g = rng.choice(chosen)
    facts = " ".join(r.render_fact(a, "a0", origins[a], key=f"{a}|{idx}|{rng.random()}") for a in chosen)
    q = r.render_query("recall", entity=g, attribute="a0")
    return Example(f"{facts} {q}", _render_answer(origins[g]), length)


def _ex_binding(spec, w, r, oracle, rng, length):
    """Last-write-wins binding (no recall): resolve the current holder of an object, answer the agent."""
    objs = list(w.objects[:spec.n_objects_active])
    ev = [Event("give", (rng.choice(objs), rng.choice(w.agents))) for _ in range(length)]
    obj = rng.choice(sorted({e.args[0] for e in ev}))
    holder = oracle.easy_holder(ev, obj)
    hist = " ".join(r.render_history(tuple(ev), with_steps=True))
    q = r.render_query("state_easy", target=obj)
    return Example(f"{hist} {q}", _render_answer(holder), length, {"obj": obj})


def _ex_composite(spec, w, r, oracle, fixed_origins, rng, length, idx):
    """binding × recall: resolve holder (oracle) then recall its a0. memorized or in-context-copy recall."""
    pool = spec.recall_pool or spec.k
    chosen = list(w.agents[:pool]) if spec.memorized_recall else rng.sample(list(w.agents), pool)
    origins = fixed_origins if spec.memorized_recall else dict(zip(chosen, rng.sample(list(w.value_vocab), pool)))
    objs = list(w.objects[:spec.n_objects_active])
    ev = [Event("give", (rng.choice(objs), rng.choice(chosen))) for _ in range(length)]
    obj = rng.choice(sorted({e.args[0] for e in ev}))
    holder = oracle.easy_holder(ev, obj)                          # gold via the oracle
    value = origins[holder]
    facts = " ".join(r.render_fact(a, "a0", origins[a], key=f"{a}|{idx}|{rng.random()}") for a in chosen)
    hist = " ".join(r.render_history(tuple(ev), with_steps=True))
    q = r.render_query("recall", attribute="a0", entity=f"the holder of {obj}")
    prompt = f"{facts} {hist} {q}"
    meta = {"holder": holder, "obj": obj}
    if spec.worked_trace:    # oracle worked-trace = optional TRAINING signal, not part of the scored answer
        meta["trace"] = " ".join(oracle.easy_holder(ev, obj, t=t) for t in range(1, length + 1))
    return Example(prompt, _render_answer(f"{holder} {value}"), length, meta)


def _ex_s5(spec, w, r, oracle, rng, length, idx):
    """Sₖ state tracking: role of an agent after a swap/cycle history (worked-trace = the role trajectory)."""
    events = w.sample_hard_chain(length, episode_seed=f"{spec.name}|{idx}")
    agent = rng.choice(w.agents)
    role = oracle.hard_role(events, agent)
    hist = " ".join(r.render_history(tuple(events), with_steps=True))
    q = r.render_query("state_hard", target=agent)
    prompt = f"{hist} {q}"
    meta = {}
    if spec.worked_trace:    # oracle role-trajectory = optional TRAINING signal, not the scored answer
        meta["trace"] = " ".join(oracle.hard_role(events, agent, t=t) for t in range(1, length + 1))
    return Example(prompt, _render_answer(role), length, meta)


def _ex_conflict(spec, w, r, pmap, rng, length, idx):
    """In-weights ↔ in-context CONFLICT: the prompt states a value for the queried agent that DIFFERS from
    the value the model memorized (`pmap`) during training; the correct answer is the IN-CONTEXT value.
    A model that defaults to its weights answers `pmap[g]` (wrong); one that reads context answers v_ctx.
    `length` = number of facts in the prompt (the queried fact + distractors)."""
    pool = rng.sample(list(w.agents), min(length, len(w.agents)))
    g = rng.choice(pool)                                              # queried agent (NOT positionally fixed)
    v_ctx = rng.choice([v for v in w.value_vocab if v != pmap[g]])     # in-context value, ≠ the memorized one
    ctx = {a: (v_ctx if a == g else rng.choice(list(w.value_vocab))) for a in pool}
    present = pool[:]; rng.shuffle(present)                           # scramble order: no first-position shortcut
    facts = " ".join(r.render_fact(a, "a0", ctx[a], key=f"{a}|{idx}|{rng.random()}") for a in present)
    q = r.render_query("recall", entity=g, attribute="a0")
    return Example(f"{facts} {q}", _render_answer(v_ctx), length,
                   {"param_value": pmap[g], "in_context_value": v_ctx})


def _ex_chain(spec, w, r, rng, depth, idx):
    """Depth-k pointer chase (composition DEPTH is the difficulty axis — the knob that stays hard as
    models scale). A pointer map `nxt` (a single random cycle over `k` agents, resampled per example so
    it is genuinely in-context) is presented as a0-facts pointing agent→agent; the query nests `a0 of`
    `depth` times. Gold = nxt^depth(start). Facts are presented in scrambled order so adjacency does not
    leak the chain, and the cycle has no fixed points so every hop is load-bearing for any depth<k.
    `length` is reinterpreted as the chain DEPTH for this family."""
    cyc = rng.sample(list(w.agents), spec.k)                          # the hidden cycle order
    nxt = {cyc[i]: cyc[(i + 1) % len(cyc)] for i in range(len(cyc))}  # single k-cycle (no fixed point)
    present = cyc[:]; rng.shuffle(present)                            # render facts in scrambled order
    facts = " ".join(r.render_fact(a, "a0", nxt[a], key=f"{a}|{idx}|{rng.random()}") for a in present)
    start = rng.choice(cyc)
    cur = start
    for _ in range(depth):
        cur = nxt[cur]
    query = "what is " + "a0 of " * depth + f"{start}?"
    return Example(f"{facts} {query}", _render_answer(cur), depth, {"depth": depth, "start": start})


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------
def generate(spec: TaskSpec, split: str, n: int = 1000, length: int | None = None) -> list[Example]:
    """Deterministic examples. split in {'train','test'}. train mixes train_lengths; test uses `length`
    (one OOD/ID coordinate). Same (spec,split,length,idx) -> identical example, forever."""
    assert split in ("train", "test")
    w, r, oracle = _world(spec)
    fixed = _fixed_origins(spec, w)

    if spec.family == "conflict":     # special train protocol: reinforce the in-weights map, then conflict
        pmap = _param_map(spec, w)
        if split == "train":
            out, agents = [], list(w.agents)
            for j in range(n // 2):   # reinforce g→pmap[g] as standalone facts so the model memorizes it
                a = agents[j % len(agents)]
                out.append(Example("", f"{a} a0 {pmap[a]}.", 0, {"reinforce": True}))
            for i in range(n - n // 2):
                L = spec.train_lengths[i % len(spec.train_lengths)]
                out.append(_ex_conflict(spec, w, r, pmap, _rng(spec, "train", L, i), L, i))
            return out
        L = length or spec.eval_lengths[0]
        return [_ex_conflict(spec, w, r, pmap, _rng(spec, "test", L, i), L, i) for i in range(n)]

    lengths = spec.train_lengths if split == "train" else (length or spec.eval_lengths[0],)
    out = []
    for idx in range(n):
        rng = _rng(spec, split, lengths[idx % len(lengths)], idx)
        L = lengths[idx % len(lengths)] if split == "train" else lengths[0]
        if spec.family == "recall":
            out.append(_ex_recall(spec, w, r, fixed, rng, split, L, idx))
        elif spec.family == "binding":
            out.append(_ex_binding(spec, w, r, oracle, rng, L))
        elif spec.family == "composite":
            out.append(_ex_composite(spec, w, r, oracle, fixed, rng, L, idx))
        elif spec.family == "s5":
            out.append(_ex_s5(spec, w, r, oracle, rng, L, idx))
        elif spec.family == "chain":
            out.append(_ex_chain(spec, w, r, rng, L, idx))           # L = chain depth
        else:
            raise ValueError(spec.family)
    return out


def score_exact(pred: str, gold: str) -> int:
    """Position-strict exact match of the full answer span (a diagnostic, NOT the canonical metric).

    ``pred`` is the model's continuation after the prompt; it must match ``gold`` token-for-token over
    gold's length (extra trailing generation is ignored, so '.'-termination is not required of the
    model). The canonical headline metric is ``score_relaxed``; this is reported alongside it to expose
    pure formatting differences (e.g. a chat model emitting ``v56.`` instead of ``v56 .``)."""
    g = gold.split()
    p = pred.split()[:len(g)]
    return int(p == g)


def score_relaxed(pred: str, gold: str) -> int:
    """THE canonical metric: whitespace / trailing-period invariant match of the answer span.

    Strips a trailing period from each side and compares the first ``len(gold)`` whitespace tokens.
    This is the fair cross-regime metric: it handles API models that omit the trailing period and local
    models that emit the correct answer and then continue generating, so chat-model tokenizers that
    glue punctuation (``v56.`` vs the atomic ``v56 .``) do not change the score. See
    ``CANONICAL_METRIC``."""
    g = gold.strip().rstrip(".").split()
    p = pred.strip().rstrip(".").split()
    return int(p[: len(g)] == g)


def score_contains(pred: str, gold: str) -> int:
    """Semantic containment: every non-punctuation token in `gold` appears somewhere in `pred`.
    This is intentionally forgiving — it separates whether the model knows the answer from whether
    it guessed the exact output format."""
    g = [t for t in gold.split() if t != "."]
    p = pred.split()
    return int(all(t in p for t in g))


def score_last_n(pred: str, gold: str) -> int:
    """Match the last len(gold) tokens of `pred` to `gold` (ignoring trailing period).
    Handles common chat-model prefixes like 'The answer is ...'."""
    g = gold.strip().rstrip(".").split()
    p = pred.strip().rstrip(".").split()
    if len(p) < len(g):
        return 0
    return int(p[-len(g) :] == g)


def content_tokens(text: str) -> list[str]:
    """Normalized answer tokens with punctuation stripped: the semantic span.

    Used by the composition decomposition (holder leg vs value leg) and by trace
    scoring, so both ignore attached/legacy punctuation.
    """
    from .render import Renderer
    return [t for t in Renderer.normalize(text).split() if t != "."]


def decompose_composite(pred: str, gold: str) -> dict:
    """Per-leg accuracy for a 2-content-token composite answer (holder, value).

    Returns:
        holder_ok: first content token matches (the state-tracking / binding leg)
        value_ok:  second content token matches, among 2-token answers (the recall leg)
        prefix:    longest matching prefix length (0/1/2) — a direct read of where
                   composition breaks (0=neither, 1=holder-only, 2=both)
    """
    g = content_tokens(gold)
    p = content_tokens(pred)
    k = 0
    while k < len(g) and k < len(p) and p[k] == g[k]:
        k += 1
    return {
        "holder_ok": int(len(g) >= 1 and len(p) >= 1 and p[0] == g[0]),
        "value_ok":  int(len(g) >= 2 and len(p) >= 2 and p[1] == g[1]),
        "prefix":    k,
    }


def trace_accuracy(pred_trace: str, gold_trace: str) -> dict:
    """Token-level agreement of a self-generated/unrolled trace against the oracle trajectory.

    Used by the autoregressive experiments (E3): for composite/s5 tasks the gold
    `meta["trace"]` is the oracle's per-step state (holder / role). This scores how
    far a model's self-produced trace follows the correct trajectory.

    Returns:
        token_acc: fraction of gold trace tokens matched at the right position
        first_diverge: index of the first mismatched token (len(gold) if all match)
        full_match: token_acc == 1.0
    """
    g = content_tokens(gold_trace)
    p = content_tokens(pred_trace)
    if not g:
        return {"token_acc": 1.0, "first_diverge": 0, "full_match": True}
    matched = 0
    first_div = len(g)
    for i, gt in enumerate(g):
        pt = p[i] if i < len(p) else None
        if pt == gt:
            matched += 1
        else:
            first_div = min(first_div, i)
            break
    return {
        "token_acc": matched / len(g),
        "first_diverge": first_div,
        "full_match": matched == len(g),
    }


# canonical frozen reference instances (scale via .scaled(...)). `kind` separates scored benchmark tasks
# from controls/experimental tasks (see the kind field: benchmark|control|experimental).
CANONICAL = {
    # control: memorized 5-entry map shared train/test -> in-weights lookup, not retrieval. Use as a
    # positive control / floor-check, not a recall score. The honest recall task is recall_copy_v1.
    "recall_v1":        TaskSpec("recall_v1", "recall", k=5, kind="control"),
    # genuine in-context-copy recall (random map, no memorization). `length` grows the distractor POOL
    # (1-of-pool), so eval_lengths is a real recall-extrapolation axis. Like chain_v1 this separates the
    # learnable regime from the cliff: a learnability probe found in-context recall solvable at small pool
    # (pool 5: hybrid 1.00 vs transformer 0.19, §4) but flooring as the pool grows (pool 16 ≈ 0.28, pool 64
    # ≈ 0.01) — a binding-load cliff. So the scored default trains at learnable pools (2–5) and evaluates
    # the binding-load extrapolation (pools 6, 8); the harder large-pool regime is a .scaled(k=…) variant.
    "recall_copy_v1":   TaskSpec("recall_copy_v1", "recall", k=8, memorized_recall=False,
                                 value_vocab_size=64, train_lengths=(2, 3, 4, 5), eval_lengths=(6, 8)),
    # NOTE: last-write-wins binding *is* the delta-rule update, so delta-rule recurrences have a structural
    # prior here — this measures last-write-wins tracking, not a neutral cross-architecture state score.
    "binding_v1":       TaskSpec("binding_v1", "binding", n_objects_active=4),
    # control: the recall leg is the memorized 5-map -> this isolates the BINDING leg (recall saturated),
    # it is not a composition score. The real composition task is composite_copy_v1.
    "composite_v1":     TaskSpec("composite_v1", "composite", k=5, memorized_recall=True, kind="control"),
    # the flagship: genuine 2-hop binding × in-context-copy. BIMODAL at the emergence threshold -> report
    # p(converge) over >=5 seeds, not a mean.
    "composite_copy_v1": TaskSpec("composite_copy_v1", "composite", k=32, recall_pool=16,
                                  memorized_recall=False, value_vocab_size=128,
                                  train_lengths=(4, 8, 16), eval_lengths=(16, 32, 64)),
    # the EXACT §5 scale-experiment difficulty point, registered for one-command reproducibility: a small
    # recall pool (k=5, 1-of-5) so the recall leg is independently learnable (§4 cliff) and a composite floor
    # is attributable to composition, not recall capacity. This is what scale_wall2.py/scale_confirm.py run;
    # kind=experimental so it stays out of REPORTED (the shipped scored default is the harder composite_copy_v1).
    "composite_copy_scale_v1": TaskSpec("composite_copy_scale_v1", "composite", k=5, recall_pool=5,
                                        memorized_recall=False, value_vocab_size=128, kind="experimental",
                                        train_lengths=(4, 8, 16), eval_lengths=(16, 64)),
    # experimental: correct non-abelian S5 construct, but not reliably trainable in this harness (answer-only
    # floors in-distribution; worked-trace learns train length but compounds at generation). Needs the
    # dense-per-step regime before it is a scored task. Excluded from REPORTED.
    "s5_v1":            TaskSpec("s5_v1", "s5", k=5, worked_trace=True, kind="experimental",
                                 train_lengths=(8, 16, 32), eval_lengths=(32, 64, 128)),
    # in-weights ↔ in-context CONFLICT: the model memorizes a fixed agent→value map (reinforced in train),
    # then must OVERRIDE it from a contradicting in-context fact. Operationalizes the parametric-vs-in-context
    # axis as a measured construct: answer=in-context value; a weight-defaulting model answers the memorized
    # one. `length` = #facts in the prompt (distractor pool).
    # conflict couples in-context recall with overriding a memorized map, so it inherits recall's
    # binding-load cliff. With focused small-pool training the override IS cleanly solved (gdp 1.00/0.935 at
    # pool 2/3, vs transformer 0.53) and decays as the pool grows (0.40/0.31 at pool 4/5) — a binding-load
    # extrapolation axis. Centered on the learnable edge like recall_copy_v1/chain_v1.
    "conflict_v1":      TaskSpec("conflict_v1", "conflict", k=16, value_vocab_size=64,
                                 train_lengths=(2, 3), eval_lengths=(4, 5)),
    # composition DEPTH: a depth-k pointer chase where `length` is the chain depth — train shallow, eval
    # deeper. Depth is the axis that STAYS hard as models grow (the composition-depth axis). Two knobs are
    # deliberately separated: POOL SIZE k = binding load (a cliff — a learnability probe found k=6 trains
    # to ~1.0 in-distribution at the baseline scale while k=16 floors even in-distribution, like the
    # n_objects_active cliff), and DEPTH = composition (a sharp extrapolation cliff: in-distribution depths
    # solve, depth+1 floors). The scored default fixes binding low (k=6) so depth is read cleanly; the
    # harder, scale-gated k>=16 variant is available via .scaled(k=16). Depths stay < k so the cycle never
    # wraps to a recency shortcut (validity gate confirms majority/recency at floor).
    "chain_v1":         TaskSpec("chain_v1", "chain", k=6, train_lengths=(2, 3), eval_lengths=(4, 5)),
    # experimental: binding under a LARGER working set (m=8 active objects) to expose the interference
    # cliff a load probe found (m>=8 floors). Flagged not-yet-a-score: needs the dense regime + a
    # removal/overwrite primitive (not in the Event model) before it discriminates rather than just floors.
    "binding_load_v1":  TaskSpec("binding_load_v1", "binding", n_objects_active=8, kind="experimental"),
}

# the scored benchmark set (controls + experimental tasks excluded from headline reporting)
REPORTED = tuple(name for name, spec in CANONICAL.items() if spec.kind == "benchmark")


if __name__ == "__main__":  # self-test: every canonical task generates + round-trips through the oracle
    for name, spec in CANONICAL.items():
        tr = generate(spec, "train", n=50)
        te = generate(spec, "test", n=50, length=spec.eval_lengths[-1])
        assert len({e.prompt for e in tr}) > 1 and all(e.answer for e in te)
        # determinism: regenerating gives identical examples
        assert generate(spec, "test", n=5, length=spec.eval_lengths[-1])[0].prompt == \
               generate(spec, "test", n=5, length=spec.eval_lengths[-1])[0].prompt
        ex = te[0]
        print(f"{name:<18} train={len(tr)} test@{spec.eval_lengths[-1]}={len(te)}  "
              f"e.g. ...{ex.prompt[-44:]!r} -> {ex.answer!r}")
    # metric sanity
    assert score_exact("g3 v9 . extra", "g3 v9 .") == 1 and score_exact("g3 v8 .", "g3 v9 .") == 0
    print("tasks self-test OK")

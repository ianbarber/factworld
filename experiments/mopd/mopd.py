"""MOPD shared infrastructure — the reusable body for the FactWorld MOPD study.

MOPD (Multi-teacher On-Policy Distillation, arXiv:2606.30406) integrates several
RL-specialised teachers into one student by having the student roll out, routing
each rollout to its domain teacher, and updating the student on the per-token
reverse KL toward the dispatched teacher. This module holds the pieces every stage
script shares, so the stage scripts (`stage1_base`, `stage2_teachers`,
`stage3_mopd`, `evaluate`) stay thin:

  * a single shared World -> one Tokenizer covering both teacher domains + recall,
    so base / teachers / student all speak the same vocab (the same-origin invariant
    the paper shows is load-bearing for stable distillation);
  * checkpoint save/load (the repo has none — models are normally kept in memory);
  * on-policy rollout (`sample_completions`), verifiable reward (`score_relaxed`),
    a GRPO update (adapted from `phases/02-non-abelian-state/rl_grpo.py`), and the
    two MOPD distillation losses — policy-gradient (paper eq. 4) and exact
    full-vocabulary reverse KL (our low-variance analogue of the paper's top-k form,
    exact here because the atomic vocab is tiny);
  * domain routing + normalised-score eval helpers.

Model dims default to a deliberately WEAK transformer base (d=256, 4 layers) so
there is real RL headroom; `gdp_hybrid` is the recurrent fallback. The transformer
is pure PyTorch, so the self-check at the bottom runs on CPU.

  .venv/bin/python experiments/mopd/mopd.py    # self-check (CPU-ok): losses finite, ckpt round-trip
"""
from __future__ import annotations

import math
import os
import random
import statistics
import sys
from collections import defaultdict
from typing import Any

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from factworld import tasks as TK
from factworld.render import Renderer
from factworld.tokenizer import Tokenizer

_norm = Renderer.normalize     # detaches attached punctuation ("g4." -> "g4 .") before scoring

# --------------------------------------------------------------------------- #
# config
# --------------------------------------------------------------------------- #
DEVICE = "cuda"
CKPT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ckpts")

# The base architecture + size. Weak-on-purpose transformer; gdp_hybrid is the fallback.
DIMS = {"arch": "transformer", "d_model": 256, "n_layers": 4, "n_heads": 4, "d_ff": 1024}

# One shared world for every domain so a single tokenizer covers them all and every
# stage's model is vocab-compatible. _world(spec) depends only on (seed, value_vocab_size,
# n_objects, k) — family does not affect the world — so binding/chain/recall built with
# these knobs are the *same* world (same agents / objects / value vocab).
WORLD = {"seed": 0, "value_vocab_size": 64, "n_objects": 8, "k": 8}

MAX_NEW = 4          # answers are a single content token (+ optional '.') for both domains
TEMP = 1.0
ENT_COEF = 0.01      # GRPO entropy bonus (from rl_grpo.py)
ADV_CLIP = 5.0       # MOPD policy-gradient two-side advantage clip (paper default)


# --------------------------------------------------------------------------- #
# task specs (shared world) + domains
# --------------------------------------------------------------------------- #
def _shared(spec: TK.TaskSpec) -> TK.TaskSpec:
    """Pin a spec onto the shared world so it is tokenizer-compatible with the others."""
    return spec.scaled(seed=WORLD["seed"], value_vocab_size=WORLD["value_vocab_size"],
                       n_objects=WORLD["n_objects"], k=WORLD["k"])


def binding_spec(train_lengths=(4, 8, 16), eval_lengths=(16, 32, 64),
                 n_objects_active=4) -> TK.TaskSpec:
    """The last-write-wins state leg. `length` = give-stream length (the horizon axis)."""
    return _shared(TK.CANONICAL["binding_v1"]).scaled(
        n_objects_active=n_objects_active, train_lengths=train_lengths, eval_lengths=eval_lengths)


def chain_spec(train_lengths=(2, 3), eval_lengths=(3, 4, 5)) -> TK.TaskSpec:
    """The depth-k pointer-chase leg. `length` = chain DEPTH (the composition axis)."""
    return _shared(TK.CANONICAL["chain_v1"]).scaled(
        train_lengths=train_lengths, eval_lengths=eval_lengths)


def recall_spec(train_lengths=(2, 3, 4), eval_lengths=(3, 4, 5)) -> TK.TaskSpec:
    """In-context associative recall (1-of-N copy). `length` = distractor-pool size.

    A genuinely distinct computation from binding (content-addressable retrieval vs
    positional last-write-wins state). Train/eval pool sizes are centred on the base's
    RL-improvable band (small pools, where the weak base has a partial circuit)."""
    return _shared(TK.CANONICAL["recall_copy_v1"]).scaled(
        train_lengths=train_lengths, eval_lengths=eval_lengths)


# The two RL-teacher domains MOPD recombines: last-write-wins STATE (binding) and
# in-context RECALL. (chain_spec is retained above but is NOT a teacher domain — it is
# an empirically-confirmed RL wall for this instrument: the weak base is at the random
# floor even in-distribution and GRPO only memorises the training pool. See README §3.)
TEACHER_DOMAINS = {"binding": binding_spec, "recall": recall_spec}


def shared_tokenizer() -> tuple[Tokenizer, Any, Any]:
    """Build the one tokenizer + (world, renderer) covering every domain.

    Returns:
        A ``(tokenizer, world, renderer)`` triple. The world is shared across
        binding/chain/recall, so the tokenizer covers all their tokens without ``<unk>``.
    """
    w, r = TK.build_world(binding_spec())            # same world as chain/recall (shared knobs)
    return Tokenizer.build([w], r), w, r


# --------------------------------------------------------------------------- #
# data
# --------------------------------------------------------------------------- #
def doc_of(ex: TK.Example) -> str:
    """One training string for LM pretraining: ``prompt + ' ' + answer`` (prompts end in '?')."""
    return f"{ex.prompt} {ex.answer}"


def prompt_pool(spec: TK.TaskSpec, n: int, split: str = "train") -> dict[int, list[TK.Example]]:
    """A pool of examples bucketed by length, for on-policy rollout sampling.

    Args:
        spec: the task spec.
        n: total examples to draw.
        split: 'train' (mixes ``train_lengths``) or 'test' (one held-out length each).

    Returns:
        ``{length: [Example, ...]}``. Deterministic given (spec, split).
    """
    buckets: dict[int, list[TK.Example]] = defaultdict(list)
    if split == "train":
        for ex in TK.generate(spec, "train", n=n):
            buckets[ex.length].append(ex)
    else:
        for L in spec.eval_lengths:
            for ex in TK.generate(spec, "test", n=n, length=L):
                buckets[L].append(ex)
    return dict(buckets)


class FreshSampler:
    """Draw examples WITHOUT replacement from a length-bucketed pool, per length.

    On-policy RL must not reuse the same prompts every step — a fixed reused pool lets
    the policy MEMORISE prompt->answer (reward climbs on train prompts while held-out
    accuracy stays at floor, which is exactly the chain failure mode we hit). This yields
    fresh prompts each draw and only reshuffles a length's bucket once it is exhausted, so
    a run of ``steps*prompts`` draws sees near-unique prompts when the pool is large enough.
    """

    def __init__(self, spec: TK.TaskSpec, n: int, seed: int = 0):
        self.buckets = prompt_pool(spec, n, "train")
        self.lengths = list(self.buckets)
        self._rng = random.Random(seed)
        self._order = {L: self._shuffled(L) for L in self.lengths}
        self._cur = {L: 0 for L in self.lengths}

    def _shuffled(self, L: int) -> list[int]:
        idx = list(range(len(self.buckets[L])))
        self._rng.shuffle(idx)
        return idx

    def draw(self, k: int) -> list[TK.Example]:
        """Draw ``k`` fresh examples at a randomly chosen length (all equal length)."""
        L = self._rng.choice(self.lengths)
        bucket, order = self.buckets[L], self._order[L]
        out = []
        for _ in range(k):
            if self._cur[L] >= len(order):
                self._order[L] = order = self._shuffled(L)   # exhausted: reshuffle for a fresh epoch
                self._cur[L] = 0
            out.append(bucket[order[self._cur[L]]])
            self._cur[L] += 1
        return out


# --------------------------------------------------------------------------- #
# model / checkpoint I/O
# --------------------------------------------------------------------------- #
def build_fresh(tok: Tokenizer, dims: dict | None = None, device: str = DEVICE) -> Any:
    """Build a fresh model on ``device`` with the given (or default) dims."""
    from factworld.models import build_model
    d = dims or DIMS
    m = build_model(d["arch"], tok.vocab_size, d_model=d["d_model"], n_layers=d["n_layers"],
                    n_heads=d["n_heads"], d_ff=d["d_ff"])
    return m.to(device)


def save_ckpt(path: str, model: Any, tok: Tokenizer, dims: dict, meta: dict | None = None) -> None:
    """Save state_dict + dims + the tokenizer's id map (so a checkpoint is self-describing)."""
    import torch
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({"dims": dims, "vocab_size": tok.vocab_size, "token_to_id": tok.token_to_id,
                "state_dict": model.state_dict(), "meta": meta or {}}, path)


def load_ckpt(path: str, device: str = DEVICE) -> tuple[Any, Tokenizer, dict]:
    """Load a checkpoint into a fresh model. Returns ``(model, tokenizer, raw_ckpt_dict)``."""
    import torch
    ck = torch.load(path, map_location=device, weights_only=False)
    tok = Tokenizer(ck["token_to_id"])
    model = build_fresh(tok, ck["dims"], device)
    model.load_state_dict(ck["state_dict"])
    model.eval()
    return model, tok, ck


def clone_model(model: Any, tok: Tokenizer, dims: dict, device: str = DEVICE) -> Any:
    """A fresh model initialised from ``model``'s weights (used to fork teachers/student from base)."""
    import copy
    m = build_fresh(tok, dims, device)
    m.load_state_dict(copy.deepcopy(model.state_dict()))
    return m


# --------------------------------------------------------------------------- #
# on-policy rollout + reward
# --------------------------------------------------------------------------- #
def _autocast(device: str):
    import torch
    dev = "cuda" if str(device).startswith("cuda") else "cpu"
    return torch.autocast(dev, dtype=torch.bfloat16)


def sample_completions(model: Any, tok: Tokenizer, prompt_batch: list[list[int]],
                       max_new: int = MAX_NEW, temp: float = TEMP, greedy: bool = False,
                       device: str = DEVICE) -> list[list[int]]:
    """Autoregressively sample one completion per prompt. Prompts MUST be equal length.

    Equal length lets us stack without padding (all FactWorld prompts at a fixed
    domain+length tokenize to the same length), so decode is a clean batched loop.

    Args:
        prompt_batch: list of equal-length prompt-id lists.
        greedy: argmax decode (eval); otherwise temperature sampling (rollouts).

    Returns:
        One completion-id list per prompt (excludes the prompt; stops after '.').
    """
    import torch
    assert prompt_batch, "empty prompt batch"
    P = len(prompt_batch[0])
    assert all(len(p) == P for p in prompt_batch), "sample_completions requires equal-length prompts"
    cur = torch.tensor(prompt_batch, device=device)
    B = cur.size(0)
    comps: list[list[int]] = [[] for _ in range(B)]
    done = [False] * B
    model.eval()
    with torch.no_grad():
        for _ in range(max_new):
            with _autocast(device):
                logits = model(cur)[:, -1].float()
            if greedy:
                nxt = logits.argmax(-1, keepdim=True)
            else:
                nxt = torch.multinomial(torch.softmax(logits / temp, dim=-1), 1)
            cur = torch.cat([cur, nxt], dim=1)
            for b in range(B):
                if done[b]:
                    continue
                t = int(nxt[b, 0])
                comps[b].append(t)
                # answers are attached-period tokens ("g4.") — terminate on any '.'-final token.
                if tok.id_to_token.get(t, "").endswith("."):
                    done[b] = True
            if all(done):
                break
    return comps


def reward(comp: list[int], tok: Tokenizer, gold: str) -> float:
    """Verifiable 0/1 reward: relaxed-match the decoded completion against the gold answer.

    Both sides are normalised (attached punctuation detached) exactly as
    ``runner.evaluate_task`` does, so an attached-period answer token (``g4.``)
    matches the gold ``g4.`` — the canonical scoring path.
    """
    return float(TK.score_relaxed(_norm(tok.decode(comp)), _norm(gold)))


# --------------------------------------------------------------------------- #
# GRPO (Stage 2 — produce the per-domain RL teachers)
# --------------------------------------------------------------------------- #
def grpo_loss(model: Any, tok: Tokenizer, prompt_ids: list[int], comps: list[list[int]],
              adv: list[float], device: str = DEVICE, ent_coef: float = ENT_COEF) -> Any:
    """Group policy-gradient loss: ``-sum_t logp(comp_t)*adv`` + entropy bonus (adapted from rl_grpo)."""
    import torch
    import torch.nn.functional as F
    seqs = [prompt_ids + c for c in comps]
    ml = max(len(s) for s in seqs)
    inp = torch.full((len(seqs), ml), tok.pad_id, dtype=torch.long, device=device)
    for i, s in enumerate(seqs):
        inp[i, : len(s)] = torch.tensor(s, device=device)
    P = len(prompt_ids)
    with _autocast(device):
        logits = model(inp)[:, :-1].float()
    logp = F.log_softmax(logits, dim=-1)
    ent = -(logp.exp() * logp).sum(-1)
    total = ent_total = 0.0
    ntok = 0
    for i, c in enumerate(comps):
        for j in range(len(c)):
            pos = P + j - 1                       # comp token j predicted from position P+j-1
            total = total - logp[i, pos, c[j]] * adv[i]
            ent_total = ent_total + ent[i, pos]
            ntok += 1
    return (total - ent_coef * ent_total) / max(1, ntok)


def grpo_train(model: Any, tok: Tokenizer, spec: TK.TaskSpec, *, steps: int, prompts_per_step: int,
               group: int, lr: float, pool_n: int = 30000, seed: int = 0, device: str = DEVICE,
               log_every: int = 50, ent_coef: float = ENT_COEF) -> dict:
    """GRPO from a warm model (mutated in place). Returns a reward trajectory.

    Per step: draw ``prompts_per_step`` FRESH prompts (no-replacement, to prevent
    memorisation), roll out ``group`` completions each, compute group-normalised
    advantages, and accumulate the PG loss (skipping groups with zero reward variance —
    no learning signal). Reward here is on the freshly-drawn prompts, so a rising trend
    reflects generalisation, not memorised train prompts.
    """
    import torch
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    sampler = FreshSampler(spec, pool_n, seed=9000 + seed)
    traj: list[tuple[int, float]] = []
    running: list[float] = []
    for step in range(steps):
        batch = sampler.draw(prompts_per_step)
        opt.zero_grad()
        step_rew: list[float] = []
        for ex in batch:
            pids = tok.encode(ex.prompt)
            comps = sample_completions(model, tok, [pids] * group, device=device)
            rew = [reward(c, tok, ex.answer) for c in comps]
            step_rew.extend(rew)
            m = sum(rew) / len(rew)
            sd = statistics.pstdev(rew) if len(set(rew)) > 1 else 0.0
            if sd == 0:
                continue                          # no signal in this group
            adv = [(x - m) / (sd + 1e-6) for x in rew]
            model.train()
            grpo_loss(model, tok, pids, comps, adv, device, ent_coef=ent_coef).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        running.append(sum(step_rew) / max(1, len(step_rew)))
        if (step + 1) % log_every == 0:
            mr = sum(running[-log_every:]) / len(running[-log_every:])
            traj.append((step + 1, round(mr, 3)))
            print(f"    grpo[{spec.family}] step {step + 1}: mean reward {mr:.3f}", flush=True)
    return {"traj": traj}


# --------------------------------------------------------------------------- #
# MOPD distillation (Stage 3)
# --------------------------------------------------------------------------- #
def mopd_loss(student: Any, teacher: Any, tok: Tokenizer, prompts: list[list[int]],
              comps: list[list[int]], loss_form: str, device: str = DEVICE) -> tuple[Any, dict]:
    """Per-token MOPD loss over a batch routed to ONE teacher.

    Both forms distil the frozen teacher into the student along the STUDENT's own
    rollout (on-policy). ``pg`` = policy-gradient with the teacher-minus-student
    log-diff as a clipped per-token advantage (paper eq. 4). ``kl`` = exact
    full-vocabulary per-token reverse KL ``KL(student || teacher)`` (our analogue of
    the paper's top-k form; exact because the atomic vocab is small).

    Args:
        prompts: equal-or-varying-length prompt-id lists (padded here).
        comps: the student's sampled completions (one per prompt).
        loss_form: 'pg' or 'kl'.

    Returns:
        ``(loss, stats)`` where stats has mean reverse-KL and student entropy at the
        distilled answer positions (for the training-dynamics plots).
    """
    import torch
    import torch.nn.functional as F
    seqs = [p + c for p, c in zip(prompts, comps)]
    ml = max(len(s) for s in seqs)
    inp = torch.full((len(seqs), ml), tok.pad_id, dtype=torch.long, device=device)
    for i, s in enumerate(seqs):
        inp[i, : len(s)] = torch.tensor(s, device=device)
    with _autocast(device):
        s_logits = student(inp)[:, :-1].float()
    with torch.no_grad(), _autocast(device):
        t_logits = teacher(inp)[:, :-1].float()
    s_logp = F.log_softmax(s_logits, dim=-1)
    t_logp = F.log_softmax(t_logits, dim=-1)

    # answer positions: comp token j of seq i is predicted from position P_i + j - 1.
    rows, poss, toks = [], [], []
    for i, (p, c) in enumerate(zip(prompts, comps)):
        for j in range(len(c)):
            rows.append(i)
            poss.append(len(p) + j - 1)
            toks.append(c[j])
    rows = torch.tensor(rows, device=device)
    poss = torch.tensor(poss, device=device)
    toks = torch.tensor(toks, device=device)

    s_lp = s_logp[rows, poss]                     # (N, V) student log-probs at answer positions
    t_lp = t_logp[rows, poss]                     # (N, V) teacher log-probs
    rev_kl = (s_lp.exp() * (s_lp - t_lp)).sum(-1)  # KL(student || teacher) per position
    ent = -(s_lp.exp() * s_lp).sum(-1)
    if loss_form == "kl":
        loss = rev_kl.mean()
    elif loss_form == "pg":
        s_tok = s_lp[torch.arange(len(toks), device=device), toks]
        t_tok = t_lp[torch.arange(len(toks), device=device), toks]
        adv = (t_tok - s_tok).detach().clamp(-ADV_CLIP, ADV_CLIP)   # sg[logp_T - logp_S]
        loss = -(adv * s_tok).mean()
    else:
        raise ValueError(loss_form)
    return loss, {"rev_kl": float(rev_kl.mean().item()), "ent": float(ent.mean().item())}


def mopd_train(student: Any, teachers: dict[str, Any], tok: Tokenizer,
               domain_specs: dict[str, TK.TaskSpec], *, steps: int, prompts_per_domain: int,
               loss_form: str, lr: float, pool_n: int = 30000, seed: int = 0,
               device: str = DEVICE, log_every: int = 50) -> dict:
    """MOPD Stage-3 loop. Each step draws a balanced, per-domain batch at one length,
    the student rolls out (N=1), and the batch is distilled toward its routed teacher.

    Returns training-dynamics trajectories (reverse KL, entropy) per domain.
    """
    import torch
    opt = torch.optim.AdamW(student.parameters(), lr=lr, weight_decay=0.0)
    samplers = {d: FreshSampler(s, pool_n, seed=7000 + seed + i)
                for i, (d, s) in enumerate(domain_specs.items())}
    dyn: dict[str, list] = {d: [] for d in domain_specs}
    for step in range(steps):
        opt.zero_grad()
        step_stats: dict[str, dict] = {}
        for d, spec in domain_specs.items():
            batch = samplers[d].draw(prompts_per_domain)
            prompts = [tok.encode(ex.prompt) for ex in batch]        # equal length within (d, L)
            comps = sample_completions(student, tok, prompts, device=device)
            student.train()
            loss, st = mopd_loss(student, teachers[d], tok, prompts, comps, loss_form, device)
            loss.backward()                                          # accumulate across domains
            step_stats[d] = st
        torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
        opt.step()
        if (step + 1) % log_every == 0:
            for d in domain_specs:
                dyn[d].append((step + 1, round(step_stats[d]["rev_kl"], 4),
                               round(step_stats[d]["ent"], 4)))
            msg = "  ".join(f"{d}: KL={step_stats[d]['rev_kl']:.3f} H={step_stats[d]['ent']:.3f}"
                            for d in domain_specs)
            print(f"    mopd[{loss_form}] step {step + 1}: {msg}", flush=True)
    return {"dynamics": dyn}


# --------------------------------------------------------------------------- #
# evaluation + normalised score
# --------------------------------------------------------------------------- #
def accuracy(model: Any, tok: Tokenizer, spec: TK.TaskSpec, length: int, n: int = 200,
             device: str = DEVICE, batch: int = 128) -> float:
    """Greedy relaxed-match accuracy on the held-out test split at one length."""
    exs = TK.generate(spec, "test", n=n, length=length)
    ok = 0
    for i in range(0, len(exs), batch):
        chunk = exs[i:i + batch]
        prompts = [tok.encode(e.prompt) for e in chunk]              # equal length (fixed domain+length)
        comps = sample_completions(model, tok, prompts, greedy=True, device=device)
        ok += sum(reward(c, tok, e.answer) for c, e in zip(comps, chunk))
    return ok / len(exs)


def eval_domains(model: Any, tok: Tokenizer, domain_specs: dict[str, TK.TaskSpec],
                 lengths: dict[str, list[int]] | None = None, n: int = 200,
                 device: str = DEVICE) -> dict[str, dict[int, float]]:
    """Per-domain, per-length accuracy for a model. ``{domain: {length: acc}}``."""
    out: dict[str, dict[int, float]] = {}
    for d, spec in domain_specs.items():
        Ls = (lengths or {}).get(d, list(spec.eval_lengths))
        out[d] = {L: accuracy(model, tok, spec, L, n=n, device=device) for L in Ls}
    return out


def normalized_score(student_acc: float, base_acc: float, teacher_acc: float) -> float:
    """Paper's normalised score: 0 at the SFT base, 1 at the per-domain teacher."""
    denom = teacher_acc - base_acc
    if abs(denom) < 1e-9:
        return float("nan")                       # degenerate: no teacher headroom over base
    return (student_acc - base_acc) / denom


# --------------------------------------------------------------------------- #
# self-check (CPU-ok: transformer arch is pure PyTorch)
# --------------------------------------------------------------------------- #
def _selfcheck() -> None:
    """Smoke test: shared tokenizer covers all domains, losses are finite, ckpt round-trips."""
    import tempfile

    import torch

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok, _w, _r = shared_tokenizer()
    print(f"vocab_size={tok.vocab_size}")

    # no <unk> in any domain's prompts/answers
    for d, mk in {**TEACHER_DOMAINS, "recall": recall_spec}.items():
        spec = mk()
        for ex in TK.generate(spec, "test", n=8, length=spec.eval_lengths[0]):
            ids = tok.encode(doc_of(ex))
            assert tok.unk_id not in ids, f"<unk> in {d}: {doc_of(ex)!r}"
    print("tokenizer covers binding/chain/recall with no <unk>  OK")

    dims = {**DIMS, "d_model": 64, "n_layers": 2, "d_ff": 128}
    model = build_fresh(tok, dims, dev)
    teacher = clone_model(model, tok, dims, dev)

    bspec = binding_spec()
    pool = prompt_pool(bspec, 32, "train")
    L = list(pool)[0]
    prompts = [tok.encode(pool[L][i].prompt) for i in range(4)]
    comps = sample_completions(model, tok, prompts, device=dev)
    assert len(comps) == 4
    print("sample_completions OK")

    # GRPO loss finite
    pids = prompts[0]
    gcomps = sample_completions(model, tok, [pids] * 4, device=dev)
    gl = grpo_loss(model, tok, pids, gcomps, [1.0, -1.0, 0.5, -0.5], dev)
    assert torch.isfinite(gl), "grpo loss not finite"

    # MOPD losses finite + clipped
    for form in ("pg", "kl"):
        loss, st = mopd_loss(model, teacher, tok, prompts, comps, form, dev)
        assert torch.isfinite(loss), f"{form} loss not finite"
        assert st["rev_kl"] >= -1e-4, "reverse KL should be >= 0"
    print("grpo + mopd (pg/kl) losses finite  OK")

    # checkpoint round-trip
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "m.pt")
        save_ckpt(path, model, tok, dims, {"tag": "selfcheck"})
        m2, t2, ck = load_ckpt(path, dev)
        x = torch.tensor([prompts[0]], device=dev)
        with torch.no_grad():
            a = model(x).float()
            b = m2(x).float()
        assert torch.allclose(a, b, atol=1e-4), "ckpt weights differ after round-trip"
        assert t2.vocab_size == tok.vocab_size and ck["meta"]["tag"] == "selfcheck"
    print("checkpoint save/load round-trip  OK")
    print("mopd self-check PASSED")


if __name__ == "__main__":
    _selfcheck()

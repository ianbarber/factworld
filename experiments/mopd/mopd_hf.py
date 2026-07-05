"""MOPD on a pretrained Qwen3-1.7B — shared infra (LoRA adapters as base/teachers/student).

The pivot from the from-scratch study: a real pretrained model has genuine capability that
outcome-RL can specialise (the paper's own setting is a pretrained Qwen3). We use the smallest
cached Qwen3 (1.7B) on a single 3090, and realise MOPD's same-origin invariant *literally*: one
FROZEN Qwen3 backbone carries several LoRA adapters —

  * base       = adapters disabled (the pretrained backbone; the norm-score 0-anchor)
  * teacher_*  = one GRPO-trained adapter per domain (the norm-score 1-anchors)
  * student    = the MOPD-distilled adapter (forked from base = a fresh adapter)

so every model shares identical backbone weights (initial student<->teacher KL is tiny by
construction). Distillation swaps the active adapter for the teacher forward pass; because only
adapters train, base + all teachers + the student fit comfortably in 24 GB.

Domains (both PARTIAL on the base -> real RL headroom; distinct computations; no thinking, so
rollouts are short and the answer span is clean): binding (last-write-wins STATE) and recall
under a large distractor pool (associative RETRIEVAL). See bench_qwen.py for the base profile.

  .venv/bin/python experiments/mopd/mopd_hf.py    # self-check: load, adapters, rollout, losses
"""
from __future__ import annotations

import os
import random
import statistics
import sys
from typing import Any

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from factworld import tasks as TK
from factworld.render import Renderer

MODEL = "Qwen/Qwen3-1.7B"
CKPT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ckpts_hf")
INSTRUCTION = ("Read the statements and answer the question. "
               "Respond with ONLY the answer token(s) (e.g. `g4` or `g4 v56`), nothing else.")
_norm = Renderer.normalize
ADV_CLIP = 5.0
MAX_NEW = 8              # no thinking: answers are 1-2 content tokens (+ '.')

# The two RL-teacher domains (both partial on the base; see bench_qwen.py / config sweep).
DOMAINS: dict[str, Any] = {
    "binding": lambda: TK.CANONICAL["binding_v1"].scaled(
        train_lengths=(8, 16), eval_lengths=(16, 24, 32)),
    "recall": lambda: TK.CANONICAL["recall_copy_v1"].scaled(
        k=64, value_vocab_size=128, train_lengths=(12, 16), eval_lengths=(16, 24)),
}


# --------------------------------------------------------------------------- #
# prompts / reward
# --------------------------------------------------------------------------- #
def build_chat(tok: Any, task_prompt: str) -> str:
    """Chat-format a FactWorld prompt with the answer-only instruction, thinking disabled."""
    msgs = [{"role": "user", "content": f"{INSTRUCTION}\n\n{task_prompt}"}]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                   enable_thinking=False)


def reward(pred_text: str, gold: str) -> float:
    """Verifiable 0/1 reward = the canonical FactWorld relaxed match (train == eval metric)."""
    return float(TK.score_relaxed(_norm(pred_text), _norm(gold)))


def prompt_pool(spec: TK.TaskSpec, n: int) -> dict[int, list[TK.Example]]:
    """Length-bucketed fresh training examples (mixes train_lengths)."""
    from collections import defaultdict
    buckets: dict[int, list[TK.Example]] = defaultdict(list)
    for ex in TK.generate(spec, "train", n=n):
        buckets[ex.length].append(ex)
    return dict(buckets)


# --------------------------------------------------------------------------- #
# backbone + LoRA adapters
# --------------------------------------------------------------------------- #
def load_backbone(device: str = "cuda", grad_ckpt: bool = True) -> tuple[Any, Any]:
    """Load the frozen Qwen3 backbone + tokenizer (left-padded for batched generation)."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to(device)
    if grad_ckpt:
        model.config.use_cache = False
        model.gradient_checkpointing_enable()
    return model, tok


def to_peft(model: Any, first_adapter: str, r: int = 16, alpha: int = 32) -> Any:
    """Wrap the backbone in a PeftModel with an initial LoRA adapter (fresh = a 'base' fork)."""
    from peft import LoraConfig, get_peft_model
    cfg = LoraConfig(r=r, lora_alpha=alpha, lora_dropout=0.0, bias="none",
                     task_type="CAUSAL_LM", target_modules="all-linear")
    return get_peft_model(model, cfg, adapter_name=first_adapter)


def add_adapter(pmodel: Any, name: str, r: int = 16, alpha: int = 32) -> None:
    """Add another fresh LoRA adapter on the same backbone."""
    from peft import LoraConfig
    cfg = LoraConfig(r=r, lora_alpha=alpha, lora_dropout=0.0, bias="none",
                     task_type="CAUSAL_LM", target_modules="all-linear")
    pmodel.add_adapter(name, cfg)


def activate(pmodel: Any, name: str, train: bool) -> None:
    """Make ``name`` the active adapter and set requires_grad only on it."""
    pmodel.set_adapter(name)
    for pname, p in pmodel.named_parameters():
        p.requires_grad = train and (".lora_" in pname) and (f".{name}." in pname)


def save_adapter(pmodel: Any, name: str) -> None:
    """Save adapter ``name`` to ``CKPT_DIR/name/`` (peft nests selected adapters by name)."""
    pmodel.set_adapter(name)
    pmodel.save_pretrained(CKPT_DIR, selected_adapters=[name])


def load_adapter(pmodel: Any, name: str) -> None:
    """Load adapter ``name`` from ``CKPT_DIR/name/``."""
    pmodel.load_adapter(os.path.join(CKPT_DIR, name), adapter_name=name)


def adapter_exists(name: str) -> bool:
    return os.path.isfile(os.path.join(CKPT_DIR, name, "adapter_config.json"))


# --------------------------------------------------------------------------- #
# rollout
# --------------------------------------------------------------------------- #
def _strip(text: str) -> str:
    return text.split("</think>")[-1].strip() if "</think>" in text else text.strip()


def sample(pmodel: Any, tok: Any, chat: str, g: int, temp: float, greedy: bool = False,
           device: str = "cuda") -> tuple[list[list[int]], list[str], int]:
    """Sample ``g`` completions for one chat prompt with the ACTIVE adapter.

    Returns:
        (completion_id_lists, decoded_texts, prompt_len). Completion ids exclude the prompt
        and are truncated at the first EOS.
    """
    import torch
    enc = tok([chat], return_tensors="pt").to(device)
    P = enc.input_ids.shape[1]
    was_ckpt = getattr(pmodel, "is_gradient_checkpointing", False)
    pmodel.config.use_cache = True
    with torch.no_grad():
        out = pmodel.generate(**enc, max_new_tokens=MAX_NEW, do_sample=not greedy,
                              temperature=(1.0 if greedy else temp), top_p=1.0, top_k=0,
                              num_return_sequences=g, pad_token_id=tok.pad_token_id)
    pmodel.config.use_cache = not was_ckpt
    gen = out[:, P:]
    comps, texts = [], []
    for row in gen:
        ids = row.tolist()
        if tok.eos_token_id in ids:
            ids = ids[: ids.index(tok.eos_token_id) + 1]
        comps.append(ids)
        texts.append(_strip(tok.decode(ids, skip_special_tokens=True)))
    return comps, texts, P


def _seq_logp(pmodel: Any, tok: Any, prompt_ids: list[int], comps: list[list[int]],
              device: str) -> tuple[Any, Any, Any]:
    """Log-probs of each completion token under the ACTIVE adapter.

    Returns (logp_full, comp_mask, seqs): logp_full is (B, Lmax, V) log-softmax over vocab
    aligned so logp_full[b, t] predicts seqs[b, t+1]; comp_mask marks completion-token positions.
    """
    import torch
    import torch.nn.functional as F
    seqs = [prompt_ids + c for c in comps]
    Lmax = max(len(s) for s in seqs)
    pad = tok.pad_token_id
    inp = torch.full((len(seqs), Lmax), pad, dtype=torch.long, device=device)
    attn = torch.zeros((len(seqs), Lmax), dtype=torch.long, device=device)
    mask = torch.zeros((len(seqs), Lmax), dtype=torch.bool, device=device)
    P = len(prompt_ids)
    for i, s in enumerate(seqs):
        inp[i, : len(s)] = torch.tensor(s, device=device)
        attn[i, : len(s)] = 1
        mask[i, P: len(s)] = True                    # completion tokens
    logits = pmodel(input_ids=inp, attention_mask=attn).logits.float()
    logp = F.log_softmax(logits, dim=-1)
    return logp, mask, inp


# --------------------------------------------------------------------------- #
# GRPO (Stage 2 teachers)
# --------------------------------------------------------------------------- #
def grpo_train(pmodel: Any, tok: Any, adapter: str, spec: TK.TaskSpec, *, steps: int,
               prompts_per_step: int, group: int, lr: float, temp: float = 1.0,
               ent_coef: float = 0.0, pool_n: int = 20000, seed: int = 0,
               device: str = "cuda", log_every: int = 20) -> dict:
    """GRPO-train ``adapter`` on ``spec`` (verifiable reward). Fresh prompts each step."""
    import torch
    torch.manual_seed(seed)                          # reproducible rollouts (do_sample) per seed
    activate(pmodel, adapter, train=True)
    opt = torch.optim.AdamW([p for p in pmodel.parameters() if p.requires_grad], lr=lr)
    pool = prompt_pool(spec, pool_n)
    order = {L: random.Random(seed).sample(pool[L], len(pool[L])) for L in pool}
    cur = {L: 0 for L in pool}
    lengths = list(pool)
    rng = random.Random(9000 + seed)
    traj, running = [], []
    for step in range(steps):
        L = rng.choice(lengths)
        batch = []
        for _ in range(prompts_per_step):
            if cur[L] >= len(order[L]):
                order[L] = rng.sample(pool[L], len(pool[L])); cur[L] = 0
            batch.append(order[L][cur[L]]); cur[L] += 1
        opt.zero_grad()
        step_rew = []
        for ex in batch:
            chat = build_chat(tok, ex.prompt)
            activate(pmodel, adapter, train=True)
            comps, texts, _P = sample(pmodel, tok, chat, group, temp, device=device)
            rew = [reward(t, ex.answer) for t in texts]
            step_rew.extend(rew)
            m = sum(rew) / len(rew)
            sd = statistics.pstdev(rew) if len(set(rew)) > 1 else 0.0
            if sd == 0:
                continue
            adv = [(x - m) / (sd + 1e-6) for x in rew]
            pids = tok(chat).input_ids
            logp, mask, inp = _seq_logp(pmodel, tok, pids, comps, device)
            tok_lp = logp[:, :-1].gather(-1, inp[:, 1:].unsqueeze(-1)).squeeze(-1)  # logp of next tok
            m2 = mask[:, 1:].float()
            advt = torch.tensor(adv, device=device).unsqueeze(1)
            pg = -(advt * tok_lp * m2).sum() / m2.sum().clamp(min=1)
            if ent_coef:
                ent = -(logp.exp() * logp).sum(-1)[:, :-1]
                pg = pg - ent_coef * (ent * m2).sum() / m2.sum().clamp(min=1)
            pg.backward()
        torch.nn.utils.clip_grad_norm_([p for p in pmodel.parameters() if p.requires_grad], 1.0)
        opt.step()
        running.append(sum(step_rew) / max(1, len(step_rew)))
        if (step + 1) % log_every == 0:
            mr = sum(running[-log_every:]) / len(running[-log_every:])
            traj.append((step + 1, round(mr, 3)))
            print(f"    grpo[{adapter}] step {step + 1}: mean reward {mr:.3f}", flush=True)
    return {"traj": traj}


# --------------------------------------------------------------------------- #
# MOPD distillation (Stage 3)
# --------------------------------------------------------------------------- #
def mopd_train(pmodel: Any, tok: Any, student: str, teachers: dict[str, str],
               domain_specs: dict[str, TK.TaskSpec], *, steps: int, prompts_per_domain: int,
               loss_form: str, lr: float, temp: float = 1.0, pool_n: int = 20000,
               seed: int = 0, device: str = "cuda", log_every: int = 20) -> dict:
    """MOPD: distil each domain teacher adapter into the student adapter on the student's rollouts.

    ``pg`` = clipped teacher-minus-student log-diff advantage (paper eq. 4);
    ``kl`` = exact full-vocabulary per-token reverse KL (student || teacher).
    Adapter is swapped to the teacher for its (no-grad) scoring forward pass.
    """
    import torch
    torch.manual_seed(seed)                          # reproducible rollouts (do_sample) per seed
    activate(pmodel, student, train=True)
    opt = torch.optim.AdamW([p for p in pmodel.parameters() if p.requires_grad], lr=lr)
    pools = {d: prompt_pool(s, pool_n) for d, s in domain_specs.items()}
    rng = random.Random(7000 + seed)
    dyn = {d: [] for d in domain_specs}
    for step in range(steps):
        opt.zero_grad()
        stats = {}
        for d, spec in domain_specs.items():
            L = rng.choice(list(pools[d]))
            batch = rng.sample(pools[d][L], min(prompts_per_domain, len(pools[d][L])))
            kl_acc = ent_acc = 0.0
            nb = 0
            for ex in batch:
                chat = build_chat(tok, ex.prompt)
                pids = tok(chat).input_ids
                activate(pmodel, student, train=False)
                comps, _texts, _P = sample(pmodel, tok, chat, 1, temp, device=device)
                # teacher log-probs (no grad, teacher adapter)
                activate(pmodel, teachers[d], train=False)
                with torch.no_grad():
                    t_logp, mask, inp = _seq_logp(pmodel, tok, pids, comps, device)
                # student log-probs (grad, student adapter)
                activate(pmodel, student, train=True)
                s_logp, _mask, _inp = _seq_logp(pmodel, tok, pids, comps, device)
                m = mask[:, 1:].float()
                s_lp, t_lp = s_logp[:, :-1], t_logp[:, :-1]
                rev_kl = (s_lp.exp() * (s_lp - t_lp)).sum(-1)           # (B, L-1)
                if loss_form == "kl":
                    loss = (rev_kl * m).sum() / m.sum().clamp(min=1)
                elif loss_form == "pg":
                    nxt = inp[:, 1:].unsqueeze(-1)
                    s_tok = s_lp.gather(-1, nxt).squeeze(-1)
                    t_tok = t_lp.gather(-1, nxt).squeeze(-1)
                    adv = (t_tok - s_tok).detach().clamp(-ADV_CLIP, ADV_CLIP)
                    loss = -(adv * s_tok * m).sum() / m.sum().clamp(min=1)
                else:
                    raise ValueError(loss_form)
                loss.backward()
                ent = -(s_lp.exp() * s_lp).sum(-1)
                kl_acc += float((rev_kl * m).sum() / m.sum().clamp(min=1))
                ent_acc += float((ent * m).sum() / m.sum().clamp(min=1))
                nb += 1
            stats[d] = {"rev_kl": kl_acc / max(1, nb), "ent": ent_acc / max(1, nb)}
        torch.nn.utils.clip_grad_norm_([p for p in pmodel.parameters() if p.requires_grad], 1.0)
        opt.step()
        if (step + 1) % log_every == 0:
            for d in domain_specs:
                dyn[d].append((step + 1, round(stats[d]["rev_kl"], 4), round(stats[d]["ent"], 4)))
            msg = "  ".join(f"{d}: KL={stats[d]['rev_kl']:.3f} H={stats[d]['ent']:.3f}"
                            for d in domain_specs)
            print(f"    mopd[{loss_form}] step {step + 1}: {msg}", flush=True)
    return {"dynamics": dyn}


# --------------------------------------------------------------------------- #
# eval
# --------------------------------------------------------------------------- #
def accuracy(pmodel: Any, tok: Any, adapter: str | None, spec: TK.TaskSpec, length: int,
             n: int = 200, device: str = "cuda", batch: int = 32) -> float:
    """Greedy relaxed-match accuracy. adapter=None -> base backbone (adapters disabled)."""
    import torch
    exs = TK.generate(spec, "test", n=n, length=length)
    ok = 0

    def _gen(chats):
        enc = tok(chats, return_tensors="pt", padding=True).to(device)
        pmodel.config.use_cache = True
        with torch.no_grad():
            out = pmodel.generate(**enc, max_new_tokens=MAX_NEW, do_sample=False,
                                  pad_token_id=tok.pad_token_id)
        return [_strip(t) for t in tok.batch_decode(out[:, enc.input_ids.shape[1]:],
                                                    skip_special_tokens=True)]

    for i in range(0, len(exs), batch):
        chunk = exs[i:i + batch]
        chats = [build_chat(tok, e.prompt) for e in chunk]
        if adapter is None:
            with pmodel.disable_adapter():
                preds = _gen(chats)
        else:
            activate(pmodel, adapter, train=False)
            preds = _gen(chats)
        ok += sum(reward(p, e.answer) for p, e in zip(preds, chunk))
    return ok / len(exs)


def eval_all(pmodel: Any, tok: Any, adapter: str | None, domain_specs: dict[str, TK.TaskSpec],
             n: int = 200, device: str = "cuda") -> dict[str, dict[int, float]]:
    return {d: {L: accuracy(pmodel, tok, adapter, s, L, n=n, device=device)
                for L in s.eval_lengths} for d, s in domain_specs.items()}


def normalized_score(model_acc: float, base_acc: float, teacher_acc: float) -> float:
    denom = teacher_acc - base_acc
    return float("nan") if abs(denom) < 1e-9 else (model_acc - base_acc) / denom


# --------------------------------------------------------------------------- #
# self-check
# --------------------------------------------------------------------------- #
def _selfcheck() -> None:
    import torch
    if not torch.cuda.is_available():
        print("no GPU"); return
    model, tok = load_backbone()
    pmodel = to_peft(model, "student")
    add_adapter(pmodel, "teacher_binding")
    spec = DOMAINS["binding"]()
    ex = TK.generate(spec, "test", n=1, length=16)[0]
    chat = build_chat(tok, ex.prompt)

    activate(pmodel, "teacher_binding", train=True)
    comps, texts, P = sample(pmodel, tok, chat, 4, 1.0)
    print(f"sampled {len(comps)} comps; e.g. {texts[0]!r}  reward={reward(texts[0], ex.answer)}")

    # GRPO loss path
    pids = tok(chat).input_ids
    logp, mask, inp = _seq_logp(pmodel, tok, pids, comps, "cuda")
    assert torch.isfinite(logp).all()

    # MOPD losses finite
    activate(pmodel, "student", train=True)
    s_logp, m, inp2 = _seq_logp(pmodel, tok, pids, comps, "cuda")
    with torch.no_grad():
        activate(pmodel, "teacher_binding", train=False)
        t_logp, _, _ = _seq_logp(pmodel, tok, pids, comps, "cuda")
    rev_kl = (s_logp[:, :-1].exp() * (s_logp[:, :-1] - t_logp[:, :-1])).sum(-1)
    assert (rev_kl >= -1e-3).all(), "reverse KL negative"
    print(f"reverse KL (student vs teacher, both fresh): {rev_kl.mean().item():.4f} (should be ~0)")

    # base vs adapter eval on a few examples
    base = accuracy(pmodel, tok, None, spec, 16, n=16)
    print(f"base binding@16 (n=16) = {base:.3f}")
    print("mopd_hf self-check PASSED")


if __name__ == "__main__":
    _selfcheck()

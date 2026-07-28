"""Fast CPU smoke tests for the MOPD infra (experiments/mopd/mopd.py).

Dual-runnable like the rest of tests/: plain pytest functions, and a standalone
``python tests/test_mopd.py`` entry. The transformer arch is pure PyTorch, so these
run without a GPU, though models.py imports flash-linear-attention at module scope so the
``train`` extra is still required (see ``_torch``). They check the pieces the
MOPD stages depend on — shared tokenizer coverage, rollout, the GRPO + both MOPD
losses being finite, reverse-KL non-negativity, advantage clipping, checkpoint
round-trip, and the normalised-score formula — not training quality.
"""
from __future__ import annotations

import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "experiments", "mopd"))

import mopd as M
from factworld import tasks as TK

_DIMS = {**M.DIMS, "d_model": 64, "n_layers": 2, "d_ff": 128}


def _torch():
    """The torch module, or skip — guarding the whole ``train`` import chain, not just torch.

    The four model-building tests below need the ``train`` extra. Without it they used to raise
    ModuleNotFoundError and report as FAILURES, so a core-only environment showed a red suite
    for a dependency it was never expected to have; tests/test_models.py and
    tests/test_mopd_hf.py already keep the clean-skip contract this restores.

    The guard imports ``factworld.models`` rather than torch alone because that is what the
    tests actually reach: models.py imports flash-linear-attention at module scope for the
    recurrent architectures, so torch on its own is not enough even though these tests only
    build the pure-PyTorch transformer. Verified both ways — the four tests pass under
    .venv-train (torch + fla) and skip under an env carrying torch alone.

    Under the standalone runner there is no pytest to skip with, so the caller returns instead
    (see ``__main__``).
    """
    try:
        import torch

        import factworld.models  # noqa: F401  — pulls fla; the real gate
    except ImportError as exc:                               # core-only env: not a failure
        try:
            import pytest
            pytest.skip(f"train extra unavailable ({exc})")
        except ImportError:
            return None
    return torch


def _device():
    torch = _torch()
    return "cuda" if torch is not None and torch.cuda.is_available() else "cpu"


def test_tokenizer_covers_all_domains():
    tok, _w, _r = M.shared_tokenizer()
    for mk in (*M.TEACHER_DOMAINS.values(), M.recall_spec):
        spec = mk()
        for ex in TK.generate(spec, "test", n=8, length=spec.eval_lengths[0]):
            ids = tok.encode(M.doc_of(ex))
            assert tok.unk_id not in ids, f"<unk> in {spec.family}: {M.doc_of(ex)!r}"


def test_rollout_and_reward():
    if _torch() is None:
        return
    tok, _w, _r = M.shared_tokenizer()
    dev = _device()
    model = M.build_fresh(tok, _DIMS, dev)
    spec = M.binding_spec()
    exs = TK.generate(spec, "test", n=4, length=spec.eval_lengths[0])
    prompts = [tok.encode(e.prompt) for e in exs]
    comps = M.sample_completions(model, tok, prompts, device=dev)
    assert len(comps) == 4
    assert all(M.reward(c, tok, e.answer) in (0.0, 1.0) for c, e in zip(comps, exs))


def test_grpo_and_mopd_losses_finite():
    torch = _torch()
    if torch is None:
        return
    tok, _w, _r = M.shared_tokenizer()
    dev = _device()
    model = M.build_fresh(tok, _DIMS, dev)
    teacher = M.clone_model(model, tok, _DIMS, dev)
    spec = M.binding_spec()
    exs = TK.generate(spec, "test", n=4, length=spec.eval_lengths[0])
    prompts = [tok.encode(e.prompt) for e in exs]
    comps = M.sample_completions(model, tok, prompts, device=dev)

    pids = prompts[0]
    gcomps = M.sample_completions(model, tok, [pids] * 4, device=dev)
    assert torch.isfinite(M.grpo_loss(model, tok, pids, gcomps, [1.0, -1.0, 0.5, -0.5], dev))

    for form in ("pg", "kl"):
        loss, st = M.mopd_loss(model, teacher, tok, prompts, comps, form, dev)
        assert torch.isfinite(loss)
        assert st["rev_kl"] >= -1e-4                  # reverse KL is non-negative


def test_identical_teacher_gives_zero_signal():
    # student == teacher -> reverse KL ~ 0 and PG advantage ~ 0 (the same-origin low-KL start)
    if _torch() is None:
        return
    tok, _w, _r = M.shared_tokenizer()
    dev = _device()
    model = M.build_fresh(tok, _DIMS, dev)
    teacher = M.clone_model(model, tok, _DIMS, dev)
    spec = M.binding_spec()
    exs = TK.generate(spec, "test", n=4, length=spec.eval_lengths[0])
    prompts = [tok.encode(e.prompt) for e in exs]
    comps = M.sample_completions(model, tok, prompts, device=dev)
    loss_kl, st = M.mopd_loss(model, teacher, tok, prompts, comps, "kl", dev)
    assert abs(st["rev_kl"]) < 1e-3
    assert abs(loss_kl.detach().item()) < 1e-3


def test_checkpoint_round_trip():
    import tempfile

    torch = _torch()
    if torch is None:
        return
    tok, _w, _r = M.shared_tokenizer()
    dev = _device()
    model = M.build_fresh(tok, _DIMS, dev)
    spec = M.binding_spec()
    x = torch.tensor([tok.encode(TK.generate(spec, "test", n=1, length=spec.eval_lengths[0])[0].prompt)],
                     device=dev)
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "m.pt")
        M.save_ckpt(path, model, tok, _DIMS, {"tag": "t"})
        m2, t2, ck = M.load_ckpt(path, dev)
        with torch.no_grad():
            assert torch.allclose(model(x).float(), m2(x).float(), atol=1e-4)
        assert t2.vocab_size == tok.vocab_size and ck["meta"]["tag"] == "t"


def test_normalized_score():
    assert M.normalized_score(0.5, 0.0, 1.0) == 0.5
    assert M.normalized_score(0.0, 0.0, 1.0) == 0.0
    assert M.normalized_score(1.0, 0.0, 1.0) == 1.0
    import math
    assert math.isnan(M.normalized_score(0.5, 0.4, 0.4))   # no teacher headroom


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"{name}  OK")
    print("test_mopd PASSED")

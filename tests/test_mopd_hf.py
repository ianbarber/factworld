"""Light, model-free tests for the Qwen3 MOPD path (experiments/mopd/mopd_hf.py).

These avoid loading the 1.7B backbone (that is exercised by ``python experiments/mopd/mopd_hf.py``
on a GPU). They check the pieces that need no weights: domain specs build, chat formatting,
the verifiable reward (incl. the punctuation-normalisation contract), and the normalised-score
math. The GPU/peft-dependent training path is skipped unless the deps + a CUDA device are present.
"""
from __future__ import annotations

import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "experiments", "mopd"))

import mopd_hf as H
from factworld import tasks as TK


def test_domains_build_and_are_partial_configs():
    assert set(H.DOMAINS) == {"binding", "recall"}
    for d, mk in H.DOMAINS.items():
        spec = mk()
        exs = TK.generate(spec, "test", n=4, length=spec.eval_lengths[0])
        assert len(exs) == 4 and all(e.answer for e in exs)


def test_reward_normalisation_contract():
    # attached-period token vs gold; chat prefixes handled by relaxed match + normalize
    assert H.reward("g4", "g4.") == 1.0
    assert H.reward("g4.", "g4.") == 1.0
    assert H.reward("g5", "g4.") == 0.0
    assert H.reward("g4 v56", "g4 v56.") == 1.0


def test_normalized_score():
    import math
    assert H.normalized_score(0.5, 0.0, 1.0) == 0.5
    assert H.normalized_score(1.0, 0.0, 1.0) == 1.0
    assert math.isnan(H.normalized_score(0.5, 0.4, 0.4))   # no teacher headroom


def test_build_chat_is_string_with_instruction():
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(H.MODEL)
    except Exception:                                        # no transformers / offline
        import pytest
        pytest.skip("transformers/tokenizer unavailable")
    chat = H.build_chat(tok, "who is the final holder of o0?")
    assert isinstance(chat, str) and "ONLY the answer" in chat


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"{name}  OK")
            except Exception as e:  # noqa: BLE001
                print(f"{name}  SKIP/FAIL: {e}")
    print("test_mopd_hf done")

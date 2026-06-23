"""Backend/plugin layer for FactWorld evaluation.

This module provides a small, unified interface for generating text continuations
from different kinds of models:

  - LocalBackend    : a FactWorld ``HybridLM`` checkpoint (torch + local tokenizer)
  - HFBackend       : any Hugging Face ``transformers`` causal LM
  - APIBackend      : any OpenAI-compatible chat-completion endpoint
  - FunctionBackend : a plain Python callable

All backends share the same ``generate`` contract so ``factworld.runner`` can treat
them interchangeably. Imports for heavy dependencies (torch, transformers, openai)
are deferred to construction time so the module can be imported in torch-free
environments (e.g. for the ``FunctionBackend`` smoke test path).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any


class ModelBackend(ABC):
    """Abstract base class for a FactWorld generation backend."""

    @abstractmethod
    def generate(self, prompts: list[str], max_new_tokens: int, stop_at: str | None = None) -> list[str]:
        """Generate a continuation for each prompt.

        Args:
            prompts: input strings (each typically ending in ``" : "``).
            max_new_tokens: maximum number of new tokens to generate per prompt.
            stop_at: if given, generation should stop once this token/string is
                emitted (inclusive of the stop token in the returned text).

        Returns:
            One continuation string per prompt, in the same order.
        """
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Short identifier for this backend instance."""
        ...


class LocalBackend(ModelBackend):
    """Greedy-decoding backend wrapping a FactWorld ``HybridLM`` and ``Tokenizer``."""

    def __init__(
        self,
        worlds,
        arch: str,
        d_model: int = 256,
        n_layers: int = 4,
        d_ff: int | None = None,
        model: Any | None = None,
        n_heads: int = 4,
        use_forget_gate: bool = True,
        num_householder: int = 4,
        allow_neg_eigval: bool = True,
        use_short_conv: bool = False,
        resid_init: bool = False,
        seed: int = 0,
        device: str = "cuda",
    ):
        from .render import Renderer
        from .tokenizer import Tokenizer

        self.device = device
        self.arch = arch
        self.tokenizer = Tokenizer.build(worlds, Renderer())

        if model is not None:
            self.model = model
        else:
            import torch
            from .models import build_model

            d_ff = d_ff or 4 * d_model
            torch.manual_seed(seed)
            self.model = build_model(
                arch,
                self.tokenizer.vocab_size,
                d_model=d_model,
                n_layers=n_layers,
                d_ff=d_ff,
                n_heads=n_heads,
                use_forget_gate=use_forget_gate,
                num_householder=num_householder,
                allow_neg_eigval=allow_neg_eigval,
                use_short_conv=use_short_conv,
                resid_init=resid_init,
            ).to(device)
        self.model.eval()
        self._dot_id = self.tokenizer.token_to_id["."]

    def generate(self, prompts: list[str], max_new_tokens: int, stop_at: str | None = None) -> list[str]:
        import torch

        stop_id = self.tokenizer.token_to_id.get(stop_at) if stop_at is not None else None
        out: list[str] = []
        for prompt in prompts:
            ids = self.tokenizer.encode(prompt)
            start = len(ids)
            with torch.no_grad():
                for _ in range(max_new_tokens):
                    with torch.autocast(self.device, dtype=torch.bfloat16):
                        nx = int(self.model(torch.tensor([ids], device=self.device))[0, -1].float().argmax())
                    ids.append(nx)
                    if stop_id is not None and nx == stop_id:
                        break
            out.append(self.tokenizer.decode(ids[start:]))
        return out

    @property
    def name(self) -> str:
        return f"local-{self.arch}"


class HFBackend(ModelBackend):
    """Greedy-decoding backend wrapping a Hugging Face ``transformers`` model."""

    def __init__(
        self,
        model_name_or_path: str,
        device: str | None = None,
        dtype: Any | None = None,
        **model_kwargs,
    ):
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "HFBackend requires the `transformers` package. "
                "Install it with: pip install transformers"
            ) from exc

        self.model_name_or_path = model_name_or_path
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, **model_kwargs)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        if dtype is None:
            dtype = torch.bfloat16 if device == "cuda" and torch.cuda.is_bf16_supported() else "auto"
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=dtype,
            **model_kwargs,
        ).to(device)
        self.model.eval()

    def generate(self, prompts: list[str], max_new_tokens: int, stop_at: str | None = None) -> list[str]:
        import torch

        inputs = self.tokenizer(prompts, return_tensors="pt", padding=True).to(self.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        generated = outputs[:, inputs.input_ids.shape[1] :]
        texts = self.tokenizer.batch_decode(generated, skip_special_tokens=True)
        if stop_at is not None:
            texts = [
                (t[: t.find(stop_at) + len(stop_at)] if stop_at in t else t) for t in texts
            ]
        return texts

    @property
    def name(self) -> str:
        return f"hf-{self.model_name_or_path}"


class APIBackend(ModelBackend):
    """Backend wrapping an OpenAI-compatible chat-completion endpoint."""

    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        base_url: str | None = None,
        client: Any | None = None,
    ):
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "APIBackend requires the `openai` package. "
                "Install it with: pip install openai"
            ) from exc

        self.model = model
        self.client = client if client is not None else OpenAI(api_key=api_key, base_url=base_url)

    def generate(self, prompts: list[str], max_new_tokens: int, stop_at: str | None = None) -> list[str]:
        stop = [stop_at] if stop_at is not None else None
        out: list[str] = []
        for prompt in prompts:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                top_p=1,
                max_tokens=max_new_tokens,
                stop=stop,
            )
            text = response.choices[0].message.content or ""
            # OpenAI excludes the stop token from the returned text; re-attach it
            # so downstream exact-match scoring sees the same span as local backends.
            if stop_at is not None and response.choices[0].finish_reason == "stop" and not text.endswith(stop_at):
                text += stop_at
            out.append(text)
        return out

    @property
    def name(self) -> str:
        return f"api-{self.model}"


class FunctionBackend(ModelBackend):
    """Backend that delegates generation to an arbitrary Python callable."""

    def __init__(self, fn: Callable[[list[str], int, str | None], list[str]], name: str = "function"):
        self.fn = fn
        self._name = name

    def generate(self, prompts: list[str], max_new_tokens: int, stop_at: str | None = None) -> list[str]:
        return self.fn(prompts, max_new_tokens, stop_at)

    @property
    def name(self) -> str:
        return self._name

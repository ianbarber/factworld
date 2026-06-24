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

import re
import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

try:
    import openai
except ImportError:  # pragma: no cover
    openai = None

if TYPE_CHECKING:
    from .world import World


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
        worlds: list[World],
        arch: str | None = None,
        d_model: int = 256,
        n_layers: int = 4,
        d_ff: int | None = None,
        model: Any | None = None,
        tokenizer: Any | None = None,
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
        self.tokenizer = tokenizer if tokenizer is not None else Tokenizer.build(worlds, Renderer())

        if model is not None:
            self.model = model
        else:
            if arch is None:
                raise ValueError("LocalBackend requires `arch` when `model` is not supplied")
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
        label = self.arch if self.arch is not None else type(self.model).__name__
        return f"local-{label}"


class HFBackend(ModelBackend):
    """Greedy-decoding backend wrapping a Hugging Face ``transformers`` model.

    Note: this backend uses the model's own tokenizer, which will generally not
    align token-by-token with FactWorld's atomic whitespace tokenizer. It is
    intended for evaluating external pre-trained models on the *text* of the
    benchmark, not for fair comparison with ``LocalBackend``.
    """

    def __init__(
        self,
        model_name_or_path: str,
        device: str | None = None,
        dtype: Any | None = None,
        tokenizer_kwargs: dict[str, Any] | None = None,
        system_prompt: str | None = None,
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
        self.system_prompt = system_prompt
        tokenizer_kwargs = tokenizer_kwargs or {}
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, **tokenizer_kwargs)
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
        """Generate greedily and, if ``stop_at`` is given, truncate at its first occurrence.

        Because ``transformers.generate`` does not accept arbitrary string stop
        sequences, generation runs to ``max_new_tokens`` and is truncated
        post-hoc. Set ``max_new_tokens`` just above the expected answer length.
        """
        import torch

        if self.system_prompt is not None:
            prompts = [f"{self.system_prompt}\n\n{p}" for p in prompts]
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
    """Backend wrapping an OpenAI-compatible chat-completion endpoint.

    By default calls are issued concurrently via a small thread pool
    (``max_workers``). Set ``max_workers=1`` for sequential execution, or use a
    local endpoint for large ``n``.
    """

    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        base_url: str | None = None,
        client: Any | None = None,
        max_workers: int = 4,
        system_prompt: str | None = None,
        extra_body: dict[str, Any] | None = None,
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
        self.max_workers = max_workers
        self.system_prompt = system_prompt
        self.extra_body = extra_body

    def _call_one(self, prompt: str, max_new_tokens: int, stop: list[str] | None, stop_at: str | None) -> str:
        messages: list[dict[str, str]] = []
        if self.system_prompt is not None:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": prompt})

        max_retries = 5
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=0,
                    top_p=1,
                    max_tokens=max_new_tokens,
                    stop=stop,
                    extra_body=self.extra_body,
                )
                break
            except openai.RateLimitError as exc:
                if attempt == max_retries - 1:
                    raise
                headers = getattr(exc.response, "headers", {}) or {}
                # Case-insensitive header lookup.
                hdr = {k.lower(): v for k, v in (headers.items() if hasattr(headers, "items") else [])}
                if "retry-after" in hdr:
                    wait = float(hdr["retry-after"])
                elif "x-ratelimit-reset" in hdr:
                    reset_ms = int(hdr["x-ratelimit-reset"])
                    wait = max(0.5, (reset_ms / 1000.0) - time.time())
                else:
                    wait = 2 ** attempt
                time.sleep(wait)

        choices = getattr(response, "choices", None)
        if not choices:
            return ""
        choice = choices[0]
        text = choice.message.content or ""
        # OpenAI excludes the stop token from the returned text; re-attach it
        # so downstream exact-match scoring sees the same span as local backends.
        if stop_at is not None and getattr(choice, "finish_reason", None) == "stop" and not text.endswith(stop_at):
            text += stop_at
        # Chat-model tokenizers often emit "v56." while FactWorld's atomic
        # tokenizer expects "v56 .". Normalize a trailing period that is glued
        # to the preceding token so exact-match scoring is meaningful.
        text = re.sub(r"(?<=\S)\.$", " .", text)
        return text

    def generate(self, prompts: list[str], max_new_tokens: int, stop_at: str | None = None) -> list[str]:
        stop = [stop_at] if stop_at is not None else None
        if self.max_workers == 1 or len(prompts) == 1:
            return [self._call_one(p, max_new_tokens, stop, stop_at) for p in prompts]

        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            futures = [
                pool.submit(self._call_one, p, max_new_tokens, stop, stop_at)
                for p in prompts
            ]
            return [f.result() for f in futures]

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

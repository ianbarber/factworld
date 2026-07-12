"""Diagnose qwen3.7-max's scaffolded-leg emptiness (issue #17 close-out, owner-approved
2026-07-11).

The zero_budget scaffolded cell (composite_copy_v2 @L16, holder injected, contract
"Answer: <value>") recorded empty_rate 0.98 with CLEAN stops (finish=stop, ctok ~6,
rtok 0) — the model answers SOMETHING short, but the last-"Answer:"-line extractor
finds nothing. History stores only the extracted span, so the raw text is lost;
this probe re-runs 10 deterministic items (the same generator stream as the cell)
under 4 arms and RECORDS THE RAW TEXT, to classify the failure:

  baseline   — the cell's exact prompt shape (scaffold + composite-format system
               prompt + "Reply with only one line: Answer: <value>", 96-token cap).
               Reproduces the failure and captures raw replies.
  reworded   — same prompt, contract line reworded ("End your reply with exactly
               one line of the form ...") — tests contract-PHRASING interaction.
  bigcap     — baseline contract, max_new_tokens=512 — tests a (unlikely,
               finish=stop) cap effect.
  nocontract — no contract line at all, plain scaffolded ask — tests whether the
               model answers the scaffolded question correctly when nothing asks
               for the "Answer:" wrapper (refusal-of-format vs can't-do-the-task).

Every arm scores BOTH the extracted contract span (the cell's metric) and a
value-anywhere relaxed check on the raw text, so "correct answer, missing
wrapper" is distinguishable from "wrong/empty answer". ~40 calls on qwen
(~$0.03); probe-style JSONL to results/qwen_scaffold_probe/, NO history writes.

Run:
    set -a; source .env; set +a
    .venv-api/bin/python scripts/probe_qwen_scaffold.py
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts"))

from factworld import tasks as TK
from factworld.backends import APIBackend

from eval_openrouter_grid import COMPOSITE_FORMAT_PROMPT, _build_system_prompt
from experiment_autoregressive import scaffold_prompt
from run_frontier_benchmark import (
    BASE_SYSTEM_PROMPT,
    CONTRACT_LINE_VALUE,
    TASK_PROMPTS,
    extract_contract_answer,
)

MODEL = "qwen/qwen3.7-max"
PROMPT_PRICE, COMPLETION_PRICE = 1.25, 3.75  # $/M, factworld.benchmark.MODELS
TASK, LENGTH, N = "composite_copy_v2", 16, 10
OUT_DIR = os.path.join(REPO, "results", "qwen_scaffold_probe")

REWORDED_CONTRACT = ("End your reply with exactly one line of the form "
                     "'Answer: <value>' where <value> is the requested value.")

# (arm, contract_line, max_new_tokens)
ARMS = (
    ("baseline", CONTRACT_LINE_VALUE, 96),
    ("reworded", REWORDED_CONTRACT, 96),
    ("bigcap", CONTRACT_LINE_VALUE, 512),
    ("nocontract", None, 96),
)


def main():
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise SystemExit("OPENROUTER_API_KEY not set (set -a; source .env; set +a)")
    os.makedirs(OUT_DIR, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out = os.path.join(OUT_DIR, f"probe_{stamp}.jsonl")

    spec = TK.CANONICAL[TASK]
    examples = TK.generate(spec, "test", n=N, length=LENGTH)  # same stream as the cell
    scaffolded, golds, holders = [], [], []
    for e in examples:
        p = scaffold_prompt(e, spec.name)
        assert p != e.prompt, "scaffold_prompt did not inject the holder"
        gold_ct = TK.content_tokens(e.answer)
        assert len(gold_ct) >= 2, e.answer
        scaffolded.append(p)
        golds.append(gold_ct[1])          # the VALUE alone (recall|holder)
        holders.append(e.meta["holder"])

    system_prompt = _build_system_prompt(BASE_SYSTEM_PROMPT, TASK, TASK_PROMPTS)
    assert COMPOSITE_FORMAT_PROMPT in system_prompt  # the cell's exact system prompt

    total_cost = 0.0
    for arm, contract_line, cap in ARMS:
        prompts = [f"{p}\n{contract_line}" if contract_line else p for p in scaffolded]
        backend = APIBackend(
            model=MODEL, api_key=api_key, base_url="https://openrouter.ai/api/v1",
            max_workers=8, system_prompt=system_prompt, answer_mode="raw",
            extra_body={"reasoning": {"effort": "none"}}, timeout=1800.0,
        )
        raw = backend.generate(prompts, max_new_tokens=cap, stop_at=None)
        ex_meta = backend.pop_example_meta()
        call_meta = backend.pop_call_meta()
        usage = call_meta.get("usage", {})
        cost = (usage.get("prompt_tokens", 0) / 1e6 * PROMPT_PRICE
                + usage.get("completion_tokens", 0) / 1e6 * COMPLETION_PRICE)
        total_cost += cost

        rows, n_span, n_span_ok, n_value_ok = [], 0, 0, 0
        for text, gold, holder, m in zip(raw, golds, holders, ex_meta):
            span = extract_contract_answer(text or "")
            span_toks = TK.content_tokens(span or "")
            if span_toks[:1] == [holder]:   # tolerate the injected-holder echo,
                span_toks = span_toks[1:]   # exactly as the cell's scorer does
            span_ok = int(span_toks[:1] == [gold])
            raw_toks = TK.content_tokens(text or "")
            value_ok = int(gold in raw_toks)  # value ANYWHERE in the visible reply
            n_span += span is not None
            n_span_ok += span_ok
            n_value_ok += value_ok
            rows.append({"arm": arm, "gold": gold, "holder": holder,
                         "raw": text, "span": span, "span_ok": span_ok,
                         "value_anywhere": value_ok,
                         "ctok": m.get("completion_tokens"),
                         "rtok": m.get("reasoning_tokens"),
                         "finish": m.get("finish_reason")})
        rec = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "model": MODEL, "task": TASK, "length": LENGTH, "n": N,
            "arm": arm, "contract_line": contract_line, "max_new_tokens": cap,
            "system_prompt": system_prompt,
            "metrics": {"contract_rate": n_span / N, "span_relaxed": n_span_ok / N,
                        "value_anywhere_rate": n_value_ok / N},
            "usage": {**usage, "cost_usd_est": round(cost, 4)},
            "examples": rows,
        }
        with open(out, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec) + "\n")
        print(f"{arm:<11} cap={cap:<4} contract={n_span}/{N} span_ok={n_span_ok}/{N} "
              f"value_anywhere={n_value_ok}/{N} ${cost:.4f}", flush=True)
        for r in rows[:4]:
            print(f"    raw={r['raw']!r:.100} gold={r['gold']}", flush=True)
    print(f"done -> {out} (total est ${total_cost:.4f})")


if __name__ == "__main__":
    main()

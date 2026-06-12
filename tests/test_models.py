"""M6 forward/backward smoke-test (the parity/sanity check before the matrix).

Each architecture: builds, forwards a toy batch under bf16 autocast, backprops finite
gradients, and overfits a tiny fixed batch (loss must drop). The fla recurrent kernels are
bf16-only and need CUDA, so this skips cleanly without a GPU.

Run on the 3090:  .venv/bin/python tests/test_models.py
(The stdlib test runner under system python has no torch and will print "skipped".)
"""
from __future__ import annotations

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main() -> int:
    try:
        import torch
    except Exception as e:  # noqa: BLE001
        print(f"skipped (no torch: {e})")
        return 0
    if not torch.cuda.is_available():
        print("skipped (no CUDA — fla recurrent kernels need a GPU)")
        return 0

    from factworld.models import ARCHS, build_model

    dev = "cuda"
    V, B, T, d = 256, 4, 48, 128
    torch.manual_seed(0)
    x = torch.randint(0, V, (B, T), device=dev)  # one fixed batch to overfit

    for arch in ARCHS:
        torch.manual_seed(0)
        m = build_model(arch, V, d_model=d, n_layers=4, n_heads=4, d_ff=4 * d).to(dev)
        opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=0.01)
        first = last = None
        gnorm = 0.0
        for _ in range(60):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = m(x)
                loss = torch.nn.functional.cross_entropy(
                    logits[:, :-1].reshape(-1, V), x[:, 1:].reshape(-1)
                )
            opt.zero_grad()
            loss.backward()
            gnorm = float(torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0))
            opt.step()
            first = loss.item() if first is None else first
            last = loss.item()
        assert math.isfinite(last) and math.isfinite(gnorm), f"{arch}: non-finite loss/grad"
        assert last < first * 0.7, f"{arch}: loss did not drop ({first:.3f} -> {last:.3f})"
        print(f"  ok  {arch:<12} params={m.num_params()/1e6:5.2f}M  plan={m.layers_plan}  "
              f"loss {first:.3f} -> {last:.3f}")

    print(f"\n{len(ARCHS)} architectures train.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

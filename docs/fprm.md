# FPRM — Fast Parallel Recurrent Model

FPRM is a small, weight-tied recurrent architecture added to FactWorld's local model suite.  It replaces a deep stack of distinct layers with a single shared block that is applied repeatedly, making it a compact baseline for comparing recurrent depth against the hybrid/mixer models.

## Architecture

```
embedding
   ↓
[ FPRMBlock ] × n_loops   (default n_loops = n_layers)
   ↓
RMSNorm
   ↓
tied LM head
```

Each `FPRMBlock` is:

```
x = x + RoPECausalAttention(RMSNorm(CausalConv1d(x)))
x = x + SwiGLU(RMSNorm(x))
```

- **CausalConv1d**: depthwise causal 1-D convolution with kernel size 3.  The input is padded by `kernel_size-1` on the left and the output is cropped to the input length so the receptive field is strictly left-to-right.
- **RoPECausalAttention**: the same RoPE-equipped causal softmax attention used in the transformer and hybrid baselines.
- **SwiGLU**: the same feed-forward gate used by `HybridLM`.

The same block weights are reused for every loop, so parameter count is independent of `n_loops`.

## Building

`build_model` accepts `arch="fprm"` and the standard kwargs:

```python
from factworld.models import build_model

model = build_model(
    "fprm",
    vocab_size=tok.vocab_size,
    d_model=256,
    n_layers=4,      # used as n_loops when n_loops is not given
    n_heads=4,
    d_ff=1024,
    tie_head=True,
)
```

To set a different loop count than the layer count, pass `n_loops` explicitly:

```python
model = build_model("fprm", vocab_size, d_model=256, n_layers=4, n_loops=8)
```

## Example command

Train FPRM on the v2 natural-language variant of `composite_copy_v1`:

```bash
.venv-api/bin/python scripts/sweep_local_v2.py \
    --task composite_copy_v1 \
    --arch fprm \
    --d_model 256 \
    --n_layers 4 \
    --steps 5000 \
    --seeds 0 1 2
```

Or use the generic benchmark script:

```bash
.venv-api/bin/python scripts/run_benchmark.py composite_copy_v1 \
    --arch fprm --d_model 256 --n_layers 4 --steps 5000
```

(The benchmark renders clean natural language by default; no format flag is needed.)

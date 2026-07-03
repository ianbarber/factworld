# Multi-task curriculum — one model, per-task eval

mix=binding:0.25,recall:0.30,composite_p5:0.25,composite_p16:0.20 d_model=256 n_layers=4 steps=300 seeds=[0] train_n=400

## Per-task eval (mean over seeds)

| arch | binding L16 | recall p5 | recall p16 | comp p5 L16 | comp p16 L16 | p16 holder | p16 value | p16 scaffold |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |

_Compare to single-task specialists: sweep_main_* (per-task training), cliff_diag_* (pool-16 probes)._

## Mix sweep

Re-run with `--mix binding:W,recall:W,composite_p5:W,composite_p16:W` once baseline shows signal.
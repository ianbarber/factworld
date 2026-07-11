# Staged curriculum — one model, progressive difficulty, per-task eval

schedule=binding:0.5,recall_easy:0.5:10000;binding:0.25,recall_med:0.35,composite_p5:0.4:7500;binding:0.15,recall_hard:0.25,composite_p5:0.25,composite_p16:0.35:7500 d_model=768 n_layers=8 batch=128 seeds=[0, 1, 2] train_n=80000 use_trace=False

## Final per-task eval (mean over seeds)

| arch | bind L16 | recall easy | recall med | recall hard | comp p5 | comp p16 | p16 last-N | p16 holder | p16 value | p16 scaffold |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fprm | 1.00 | 0.63 | 0.39 | 0.20 | 0.38 | 0.11 | 0.00 | 1.00 | 0.11 | 0.01 |
| gdp_hybrid | 1.00 | 0.92 | 0.95 | 0.85 | 0.99 | 0.83 | 0.00 | 1.00 | 0.83 | 0.10 |
| transformer | 0.03 | 0.09 | 0.04 | 0.03 | 0.01 | 0.00 | 0.00 | 0.07 | 0.04 | 0.01 |

## Stage details

binding:0.5,recall_easy:0.5:10000;binding:0.25,recall_med:0.35,composite_p5:0.4:7500;binding:0.15,recall_hard:0.25,composite_p5:0.25,composite_p16:0.35:7500
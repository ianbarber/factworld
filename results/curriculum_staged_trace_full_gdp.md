# Staged curriculum (dense supervision / use_trace) — one model, progressive difficulty, per-task eval

schedule=binding:0.5,recall_easy:0.5:10000;binding:0.25,recall_med:0.35,composite_p5:0.4:7500;binding:0.15,recall_hard:0.25,composite_p5:0.3,composite_p16:0.3:7500 d_model=512 n_layers=8 batch=32 seeds=[0, 1, 2] train_n=8000 use_trace=True

## Final per-task eval (mean over seeds)

| arch | bind L16 | recall easy | recall med | recall hard | comp p5 | comp p16 | p16 holder | p16 value | p16 scaffold |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gdp_hybrid | 0.24 | 0.15 | 0.08 | 0.04 | 0.00 | 0.00 | 0.02 | 0.00 | 0.01 |

## Stage details

binding:0.5,recall_easy:0.5:10000;binding:0.25,recall_med:0.35,composite_p5:0.4:7500;binding:0.15,recall_hard:0.25,composite_p5:0.3,composite_p16:0.3:7500
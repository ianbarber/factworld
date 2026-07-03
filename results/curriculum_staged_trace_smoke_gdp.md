# Staged curriculum (dense supervision / use_trace) — one model, progressive difficulty, per-task eval

schedule=binding:0.5,recall_easy:0.5:120;binding:0.25,recall_med:0.35,composite_p5:0.4:90;binding:0.15,recall_hard:0.25,composite_p5:0.3,composite_p16:0.3:90 d_model=256 n_layers=4 batch=16 seeds=[0] train_n=400 use_trace=True

## Final per-task eval (mean over seeds)

| arch | bind L16 | recall easy | recall med | recall hard | comp p5 | comp p16 | p16 holder | p16 value | p16 scaffold |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gdp_hybrid | 0.07 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |

## Stage details

binding:0.5,recall_easy:0.5:120;binding:0.25,recall_med:0.35,composite_p5:0.4:90;binding:0.15,recall_hard:0.25,composite_p5:0.3,composite_p16:0.3:90
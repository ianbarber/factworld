# Staged curriculum (dense supervision / use_trace) — one model, progressive difficulty, per-task eval

schedule=binding:0.5,recall_easy:0.5:24;binding:0.25,recall_med:0.35,composite_p5:0.4:18;binding:0.15,recall_hard:0.25,composite_p5:0.3,composite_p16:0.3:18 d_model=128 n_layers=2 batch=8 seeds=[0] train_n=200 use_trace=True

## Final per-task eval (mean over seeds)

| arch | bind L16 | recall easy | recall med | recall hard | comp p5 | comp p16 | p16 holder | p16 value | p16 scaffold |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| transformer | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |

## Stage details

binding:0.5,recall_easy:0.5:24;binding:0.25,recall_med:0.35,composite_p5:0.4:18;binding:0.15,recall_hard:0.25,composite_p5:0.3,composite_p16:0.3:18
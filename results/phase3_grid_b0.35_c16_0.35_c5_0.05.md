# Staged curriculum — one model, progressive difficulty, per-task eval

schedule=binding:0.5,recall_easy:0.5:4000;binding:0.25,recall_med:0.35,composite_p5:0.4:3000;binding:0.35,recall_hard:0.25,composite_p5:0.05,composite_p16:0.35:3000 d_model=768 n_layers=8 batch=64 seeds=[0] train_n=8000 use_trace=False

## Final per-task eval (mean over seeds)

| arch | bind L16 | recall easy | recall med | recall hard | comp p5 | comp p16 | p16 holder | p16 value | p16 scaffold |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gdp_hybrid | 0.96 | 0.13 | 0.08 | 0.06 | 0.06 | 0.02 | 0.95 | 0.02 | 0.05 |

## Stage details

binding:0.5,recall_easy:0.5:4000;binding:0.25,recall_med:0.35,composite_p5:0.4:3000;binding:0.35,recall_hard:0.25,composite_p5:0.05,composite_p16:0.35:3000
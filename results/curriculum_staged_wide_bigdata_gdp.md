# Staged curriculum — one model, progressive difficulty, per-task eval

schedule=binding:0.5,recall_easy:0.5:10000;binding:0.25,recall_med:0.35,composite_p5:0.4:7500;binding:0.15,recall_hard:0.25,composite_p5:0.3,composite_p16:0.3:7500 d_model=768 n_layers=8 batch=32 seeds=[0, 1, 2] train_n=80000 use_trace=False

## Final per-task eval (mean over seeds)

| arch | bind L16 | recall easy | recall med | recall hard | comp p5 | comp p16 | p16 holder | p16 value | p16 scaffold |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gdp_hybrid | 0.53 | 0.49 | 0.44 | 0.53 | 0.12 | 0.04 | 0.56 | 0.04 | 0.02 |

## Stage details

binding:0.5,recall_easy:0.5:10000;binding:0.25,recall_med:0.35,composite_p5:0.4:7500;binding:0.15,recall_hard:0.25,composite_p5:0.3,composite_p16:0.3:7500
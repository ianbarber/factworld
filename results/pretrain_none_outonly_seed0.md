# Pretrain on dynamics + finetune on tasks

d_model=768 n_layers=8 batch=64 pretrain_steps=0 finetune_steps=25000 pretrain_n=0 finetune_n=40000
finetune_output_only=True use_staged_finetune=False

| arch | seed | bind | p5 exact | p16 exact | p16 holder | p16 value | scaffold | pre loss |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gdp_hybrid | 0 | 0.74 | 0.16 | 0.03 | 0.76 | 0.03 | 0.05 | nan |
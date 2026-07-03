# Pretrain on dynamics + finetune on tasks

d_model=768 n_layers=8 batch=64 pretrain_steps=15000 finetune_steps=10000 pretrain_n=40000 finetune_n=40000
finetune_output_only=False use_staged_finetune=True

| arch | seed | bind | p5 exact | p16 exact | p16 holder | p16 value | scaffold | pre loss |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gdp_hybrid | 0 | 0.10 | 0.16 | 0.11 | 0.11 | 0.11 | 0.00 | 0.168 |
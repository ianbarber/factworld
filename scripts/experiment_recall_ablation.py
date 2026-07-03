import os, sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import sys; sys.path.insert(0,".")
from factworld import tasks as TK, train as T
from factworld.backends import LocalBackend
from factworld.runner import evaluate_task
def docs(ex): return [f"{e.prompt} {e.answer}" for e in ex]
def run(label, spec, steps, train_n, seed):
    w,r=TK.build_world(spec); tr=TK.generate(spec,"train",n=train_n)
    tok,d,_=T.prepare(docs(tr),[],[w],renderer=r)
    run=T.run("transformer",tok,d,[],steps=steps,batch=32,d_model=256,n_layers=4,d_ff=1024,seed=seed,return_model=True,device="cuda")
    be=LocalBackend([w],arch="transformer",model=run["model"],tokenizer=tok,device="cuda")
    res={L:evaluate_task(be,spec,split="test",n=100,length=L)["overall"] for L in spec.eval_lengths}
    print(f"  {label:40s} s{seed}: pool6={res.get(6,0):.2f} pool8={res.get(8,0):.2f}",flush=True)
spec_n=TK.CANONICAL["recall_copy_v1"]
spec_w=spec_n.scaled(train_lengths=(2,3,4,5,6,7,8),eval_lengths=(6,8))
print("### transformer recall: narrow(2-5) vs wide(2-8) pools, 8k vs 20k ###",flush=True)
for s in [0,1]:
    run("narrow 8k/8k",spec_n,8000,8000,s)
for s in [0,1]:
    run("narrow 20k/20k",spec_n,20000,20000,s)
for s in [0,1]:
    run("WIDE 8k/8k",spec_w,8000,8000,s)
for s in [0,1]:
    run("WIDE 20k/20k",spec_w,20000,20000,s)

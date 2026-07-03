import os, sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import sys; sys.path.insert(0,".")
from factworld import tasks as TK, train as T
from factworld.backends import LocalBackend
from factworld.runner import evaluate_task
def docs(ex): return [f"{e.prompt} {e.answer}" for e in ex]
def run(d, spec, seed, steps=8000, train_n=8000):
    w,r=TK.build_world(spec); tr=TK.generate(spec,"train",n=train_n)
    tok,dset,_=T.prepare(docs(tr),[],[w],renderer=r)
    run=T.run("transformer",tok,dset,[],steps=steps,batch=32,d_model=d,n_layers=4,d_ff=4*d,n_heads=4,seed=seed,return_model=True,device="cuda")
    be=LocalBackend([w],arch="transformer",model=run["model"],tokenizer=tok,device="cuda")
    res={L:evaluate_task(be,spec,split="test",n=100,length=L)["overall"] for L in spec.eval_lengths}
    params=run["model"].num_params()/1e6
    print(f"  d={d:4d} ({params:5.1f}M) s{seed}: pool6={res.get(6,0):.2f} pool8={res.get(8,0):.2f}",flush=True)
spec_n=TK.CANONICAL["recall_copy_v1"]   # narrow train pools 2-5, eval 6-8
spec_w=spec_n.scaled(train_lengths=(2,3,4,5,6,7,8),eval_lengths=(6,8))
print("### transformer in-context recall vs width ###",flush=True)
print("--- narrow training pools (2-5) ---",flush=True)
for d in [256,512,1024]:
    for s in [0,1]:
        run(d,spec_n,s)
print("--- wide training pools (2-8) ---",flush=True)
for d in [256,512,1024]:
    for s in [0,1]:
        run(d,spec_w,s)

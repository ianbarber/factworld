import os, sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import sys; sys.path.insert(0,".")
import torch, torch.nn.functional as F, math
from factworld import tasks as TK, train as T
from factworld.backends import LocalBackend
from factworld.runner import evaluate_task

def train_eval(label, arch, spec, *, d_model, n_layers, steps, batch, lr, wd, seed, train_n, dropout=None):
    w,r=TK.build_world(spec); tr=TK.generate(spec,"train",n=train_n)
    tok,docs,_=T.prepare([f"{e.prompt} {e.answer}" for e in tr],[],[w],renderer=r)
    torch.manual_seed(seed)
    m=T.build_model(arch,tok.vocab_size,d_model=d_model,n_layers=n_layers,n_heads=4,d_ff=4*d_model).to("cuda")
    if dropout is not None:
        for mod in m.modules():
            if hasattr(mod,'dropout'): mod.dropout_p=getattr(mod,'dropout_p',dropout); mod.dropout=dropout
    opt=torch.optim.AdamW(m.parameters(),lr=lr,weight_decay=wd)
    gen=torch.Generator(device="cuda").manual_seed(seed); pad=tok.pad_id; ndoc=len(docs)
    def lrm(s):
        if s<1000: return (s+1)/1000
        return 0.5*(1+math.cos(math.pi*min(1,(s-1000)/max(1,steps-1000))))
    m.train()
    for step in range(steps):
        for pg in opt.param_groups: pg["lr"]=lr*lrm(step)
        start=int(torch.randint(0,max(1,ndoc-batch),(1,),generator=gen,device="cuda").item())
        chunk=docs[start:start+batch]; ml=max(len(s) for s in chunk)
        inp=torch.full((len(chunk),ml),pad,dtype=torch.long,device="cuda")
        for ri,s in enumerate(chunk): inp[ri,:len(s)]=torch.tensor(s,device="cuda")
        with torch.autocast("cuda",dtype=torch.bfloat16):
            lg=m(inp[:,:-1]); tgt=inp[:,1:]
            ce=F.cross_entropy(lg.reshape(-1,tok.vocab_size),tgt.reshape(-1),reduction="none")
            mask=(tgt!=pad).float().reshape(-1); loss=(ce*mask).sum()/mask.sum().clamp(min=1)
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(),1.0); opt.step()
    be=LocalBackend([w],arch=arch,model=m,tokenizer=tok,device="cuda")
    a5=evaluate_task(be,spec,split="test",n=100,length=5)["overall"]
    print(f"  {label:42s}: recall@pool5={a5:.2f}",flush=True)
    return a5

spec = TK.CANONICAL["recall_copy_v1"]
print("### transformer recall recipe sweep (baseline 0.19, loss plateaus skill) ###",flush=True)
# baseline
train_eval("baseline (d256 lr1e-3 wd0.01 b32 8k)", "transformer", spec, d_model=256,n_layers=4,steps=8000,batch=32,lr=1e-3,wd=0.01,seed=0,train_n=8000)
# more data
train_eval("baseline + 40k examples", "transformer", spec, d_model=256,n_layers=4,steps=8000,batch=32,lr=1e-3,wd=0.01,seed=0,train_n=40000)
# higher wd (regularization)
train_eval("baseline + wd 0.1", "transformer", spec, d_model=256,n_layers=4,steps=8000,batch=32,lr=1e-3,wd=0.1,seed=0,train_n=8000)
# higher lr
train_eval("baseline + lr 3e-3", "transformer", spec, d_model=256,n_layers=4,steps=8000,batch=32,lr=3e-3,wd=0.01,seed=0,train_n=8000)
# larger batch (more stable grads)
train_eval("baseline + batch 64", "transformer", spec, d_model=256,n_layers=4,steps=8000,batch=64,lr=1e-3,wd=0.01,seed=0,train_n=8000)

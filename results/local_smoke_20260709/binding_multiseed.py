import sys
sys.path.insert(0, "/home/ianbarber/Projects/factworld")
from factworld import tasks as TK, train as T
from factworld.render import classify
from factworld.tokenizer import Tokenizer, _SPECIALS, _STRUCTURAL_SEED
from factworld.backends import LocalBackend
from factworld.runner import evaluate_task

def _pruned(world, renderer, used):
    tokens=set()
    for a in ("entities","value_vocab","attribute_names","objects","locations","agents","roles"):
        tokens.update(getattr(world,a))
    tokens.update(f"s{i}" for i in range(256)); tokens.update(_STRUCTURAL_SEED)
    for piece in Tokenizer._probe(world,renderer):
        for tk in piece.split():
            if classify(tk) is None: tokens.add(tk)
    if renderer.natural:
        for tk in {t for t in tokens if classify(t) is not None}:
            for suf in ("'s",".","?"):
                if tk+suf in used: tokens.add(tk+suf)
    tokens.difference_update(_SPECIALS)
    t2i={t:i for i,t in enumerate(_SPECIALS)}
    for i,tk in enumerate(sorted(tokens),start=len(_SPECIALS)): t2i[tk]=i
    return Tokenizer(t2i)

def minimal_tok(spec):
    w,r=TK.build_world(spec); ex=TK.generate(spec,"train",n=4000)
    probe=[r.render_fact(w.entities[0],w.attribute_names[0],w.value_vocab[0])]
    for ws in (False,True):
        probe+=r.render_history(w.sample_easy_chain(40,"p"),with_steps=ws)
    probe+=[r.render_query("recall",entity=w.agents[0],attribute="a0"),
            r.render_query("state_easy",target=w.objects[0])]
    surf=set()
    for d in [e.prompt for e in ex]+[e.answer for e in ex]+probe: surf.update(d.split())
    return _pruned(w,r,surf)

def docs(ex,nat):
    sep=" " if nat else ""
    return [f"{e.prompt}{sep}{e.answer}" for e in ex]

def run(spec,tk,seed):
    w,r=TK.build_world(spec); tr=TK.generate(spec,"train",n=6000)
    if tk=="default": tok,d,_=T.prepare(docs(tr,spec.natural),[],[w],renderer=r)
    else: tok=minimal_tok(spec); d=[tok.encode(t,add_eos=True) for t in docs(tr,spec.natural)]; d.sort(key=len)
    run=T.run("gdp_hybrid",tok,d,[],steps=3000,batch=32,d_model=160,n_layers=4,d_ff=640,seed=seed,return_model=True,device="cuda")
    be=LocalBackend([w],arch="gdp_hybrid",model=run["model"],tokenizer=tok,device="cuda")
    return {L:evaluate_task(be,spec,split="test",n=200,length=L)["overall"] for L in spec.eval_lengths}

print("### binding_v1 (d=160,L=4,3k steps) — tokenizer bloat effect, 3 seeds")
print("    (binding is NOT bimodal, so this isolates the tokenizer)")
for cond,spec,tk in [("v1-canon",TK.CANONICAL["binding_v1"],"default"),
                     ("nat-bloat",TK.CANONICAL["binding_v1"].scaled(natural=True),"default"),
                     ("nat-min",TK.CANONICAL["binding_v1"].scaled(natural=True),"minimal")]:
    rows=[run(spec,tk,s) for s in [0,1,2]]
    import statistics
    l64=[r[64] for r in rows]
    print(f"  {cond:10s} | L64 seeds={[f'{x:.2f}' for x in l64]}  mean={statistics.mean(l64):.3f}  "
          f"(vocab {('1470' if 'bloat' in cond else '428' if 'min' in cond else '403')})")

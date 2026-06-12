"""Does the product structure supply recall, or does the attention layer? Run the attention-free
gdp_pure (and gdn_pure) on the isolated in-context recall task and compare to the hybrid's 1.00.
If the attention-free arms floor, recall is the attention layer's job, not the product recurrence."""
import os, sys
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, REPO); sys.path.insert(0, REPO+"/scripts")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
from factworld import tasks as TK
from run_benchmark import run_task
spec = TK.CANONICAL["recall_copy_v1"].scaled(eval_lengths=(2,3,4,6,8))   # pool 2..8 (match the dissociation CI)
ARMS = ["gdp_pure", "gdn_pure", "gdp_hybrid"]   # attention-free vs the hybrid (control)
for arch in ARMS:
    for s in range(3):
        acc = run_task("recall_copy_v1", spec=spec, arch=arch, d_model=320, n_layers=4, steps=8000, seed=s)
        print(f"  {arch:<11} s{s} :: " + "  ".join(f"L{L}={acc[L]:.3f}" for L in sorted(acc)), flush=True)
print("recall_attention_test done.", flush=True)

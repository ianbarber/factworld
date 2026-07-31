"""Independent check: does dropping an INTERIOR block of events beat the registered floor?

Parses the rendered surface back into events and simulates. Shares no code with validity's
floor machinery, so it is an independent attack on the floor rather than a re-run of it.
"""
import re, sys, collections
sys.path.insert(0, "/home/ianbarber/Projects/factworld")
from factworld.tasks import CANONICAL, generate
from factworld.validity import s5_bind_operative_floor, s5_bind_floors

RE_P0   = re.compile(r"(g\d+) has role (r\d+) at the start\.")
RE_B0   = re.compile(r"(g\d+) holds (o\d+) at the start\.")
RE_SWAP = re.compile(r"^swaps the roles of (g\d+) and the agent who holds (o\d+) (at this point|at the start)\.$")
RE_SWL  = re.compile(r"^swaps the roles of (g\d+) and (g\d+)\.$")
RE_GIVE = re.compile(r"^gives (o\d+) to the agent whose role (at this point|at the start) is (r\d+)\.$")
RE_GVL  = re.compile(r"^gives (o\d+) to (g\d+)\.$")
RE_QR   = re.compile(r"what role does (g\d+) have at the end\?")
RE_QH   = re.compile(r"who holds (o\d+) at the end\?")

def parse(prompt):
    P0 = dict(RE_P0.findall(prompt))
    B0 = {o: g for g, o in RE_B0.findall(prompt)}
    body = prompt[prompt.index(" s0 ") + 1:] if " s0 " in prompt else prompt
    body = RE_QR.sub("", RE_QH.sub("", body)).strip()
    parts = re.split(r"\bs(\d+) ", body)
    evs = []
    for i in range(1, len(parts) - 1, 2):
        txt = parts[i + 1].strip()
        m = RE_SWAP.match(txt)
        if m: evs.append(("swap", m.group(1), m.group(2), m.group(3) == "at this point")); continue
        m = RE_SWL.match(txt)
        if m: evs.append(("swapl", m.group(1), m.group(2), None)); continue
        m = RE_GIVE.match(txt)
        if m: evs.append(("give", m.group(1), m.group(3), m.group(2) == "at this point")); continue
        m = RE_GVL.match(txt)
        if m: evs.append(("givel", m.group(1), m.group(2), None)); continue
        return None, None, None, None
    q = RE_QR.search(prompt) or RE_QH.search(prompt)
    return P0, B0, evs, (("role", q.group(1)) if q and q.re is RE_QR else ("hold", q.group(1)) if q else None)

def simulate(P0, B0, evs, drop):
    P, B = dict(P0), dict(B0)
    inv = {r: g for g, r in P0.items()}
    for i, (kind, a, b, dyn) in enumerate(evs):
        if i in drop:
            continue
        if kind == "swapl":
            P[a], P[b] = P[b], P[a]
        elif kind == "swap":
            other = (B if dyn else B0).get(b)
            if other is not None and other != a:
                P[a], P[other] = P[other], P[a]
        elif kind == "givel":
            B[a] = b
        elif kind == "give":
            src = P if dyn else P0
            tgt = next((g for g, rr in src.items() if rr == b), None)
            if tgt is not None:
                B[a] = tgt
    return P, B

def run(name, L, n=1500, seed=4242):
    spec = CANONICAL[name]
    ex = generate(spec, "test", n=n, length=L)
    hits = collections.Counter()
    oracle = 0
    fracs = [0.0, 0.2, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9]
    for e in ex:
        P0, B0, evs, q = parse(e.prompt)
        if q is None or not evs:
            print("  !! parse failure — ABORT"); return None
        gold = e.answer.strip().rstrip(".")
        Pf, Bf = simulate(P0, B0, evs, set())
        got = (Pf if q[0] == "role" else Bf).get(q[1])
        if got == gold:
            oracle += 1
        Lc, w = len(evs), max(1, int(round(0.10 * len(evs))))
        for f in fracs:
            s = int(round(f * (Lc - w)))
            P, B = simulate(P0, B0, evs, set(range(s, s + w)))
            if (P if q[0] == "role" else B).get(q[1]) == gold:
                hits[f] += 1
    if oracle < 0.99 * len(ex):
        print(f"  !! simulator reproduces gold on only {oracle}/{len(ex)} — ABORT"); return None
    return {f: hits[f] / len(ex) for f in fracs}

for name, Ls, k in (("s5_bind_v2", (128, 192, 256), 12), ("s5_bind_local_v2", (48, 64), 6)):
    for L in Ls:
        r = run(name, L)
        if r is None:
            continue
        ex = generate(CANONICAL[name], "test", n=1500, length=L)
        fl = s5_bind_operative_floor(s5_bind_floors(ex, k=k)) or (1.0 / (k - 1))
        ch = 1.0 / (k - 1)
        bf, bv = max(r.items(), key=lambda kv: kv[1])
        print(f"{name}@{L} chance={ch:.4f} floor={fl:.4f} | "
              + " ".join(f"{f:.2f}:{v:.3f}" for f, v in sorted(r.items()))
              + f" || BEST hole@{bf:.2f}={bv:.3f} = {bv/fl:.2f}x floor, {bv/ch:.2f}x chance")

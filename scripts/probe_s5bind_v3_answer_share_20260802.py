"""HOW MUCH OF THE TRAINING LOSS IS THE ANSWER, under the bounded-pad document mix.

``train.run`` draws each batch as a CONTIGUOUS slice of the length-sorted document list and
normalises the loss by the batch's total unmasked token count:

    loss = (ce * mask).sum() / mask.sum()

so a row whose loss is masked to ONE answer token contributes one token to a denominator its
batch-mates fill with 2L each. The answer-masked copy therefore does not carry "half the document's
loss mass"; it carries its share of the BATCH's, and that share is what this probe measures.

It matters because a pad doc and its answer-masked twin are the SAME STRING and so have the SAME
LENGTH, which puts them adjacent under the sort and guarantees they share a batch.

Prints, per (mix, ratio, sort policy), the expected fraction of the summed loss that lands on answer
tokens, over the actual batch slices the sampler can draw.
"""
from __future__ import annotations

import argparse
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts"))

from factworld import tasks as TK                                          # noqa: E402
from factworld.tokenizer import Tokenizer                                  # noqa: E402
import protocol_s5bind_v3_three_cell_20260731 as P                         # noqa: E402
import experiment_s5bind_v3_three_cell_local_20260731 as E                 # noqa: E402
import experiment_s5bind_v3_bounded_pad_20260802 as B                      # noqa: E402


_ARM_CACHE: dict = {}


def arm_rows(specs, arm, n, tok, fmt):
    """``[(plain_row, pad_row, answer_row)]`` for ``n`` items of one cell, measured once."""
    key = (arm, n, fmt)
    if key in _ARM_CACHE:
        return _ARM_CACHE[key]
    ags, obs = B.slot_order(specs[arm])
    out = []
    for e in TK.generate(specs[arm], "train", n=n):
        doc = B.narrow_document(e, fmt, ags, obs)
        if doc is None:
            continue
        dl = min(len(tok.encode(doc, add_eos=True)), B.MAX_DOC_TOKENS)
        plen = len(tok.encode(doc[:-len(e.answer) - 1]))
        pl = min(len(tok.encode(f"{e.prompt} {e.answer}", add_eos=True)), B.MAX_DOC_TOKENS)
        out.append(((pl, len(tok.encode(e.prompt)), "plain"), (dl, 1, "pad"),
                    (dl, plen, "answer")))
    _ARM_CACHE[key] = out
    return out


def build(specs, weights, train_n, tok, fmt, *, answer_ratio, group_masked, plain_docs=True,
          pad_docs=True):
    """``(lengths, prompt_lens, kinds)`` for one stage's documents in the order training sees them.

    ``group_masked`` sorts by (is_masked, length) instead of length alone, which puts every
    answer-masked document in a batch with other answer-masked documents.
    """
    rows = []
    for arm, share in sorted(weights.items()):
        n = int(round(train_n * share))
        if n <= 0:
            continue
        for pr, pd, an in arm_rows(specs, arm, n, tok, fmt):
            if plain_docs:
                rows.append(pr)
            if pad_docs:
                rows.append(pd)
            rows += [an] * answer_ratio
    key = ((lambda r: (r[1] > 1, r[0])) if group_masked else (lambda r: r[0]))
    rows.sort(key=key)
    return rows


def supervised(row):
    """Unmasked target-token count for one document under ``train.run``'s masking."""
    ln, plen, _k = row
    tgt = ln - 1                                   # targets are inp[:, 1:]
    return tgt if plen <= 1 else max(1, tgt - (plen - 1))


def answer_share(rows, batch, stride=1):
    """Per-batch supervision accounting over every slice ``train.run`` can draw.

    Returns
        ``any_share``    mean fraction of a batch's supervised tokens that are ANSWER tokens,
                         from either a plain-prompt or a pad-prompt document;
        ``pad_share``    the same restricted to PAD-PROMPT answer documents — the readout the
                         bounded protocol actually needs, since a plain-prompt answer never sees
                         a pad at all;
        ``pad_pure``     fraction of batches in which those pad-prompt answer tokens are more than
                         90% of the supervision, i.e. steps that are a readout step and not a
                         tracking step with a rounding error of readout attached.
    """
    any_n, pad_n, pure, den = 0.0, 0.0, 0, 0
    for s in range(0, max(1, len(rows) - batch), stride):
        chunk = rows[s:s + batch]
        sup = [supervised(r) for r in chunk]
        d = sum(sup)
        if not d:
            continue
        a = sum(v for v, r in zip(sup, chunk) if r[2] in ("answer", "plain"))
        p = sum(v for v, r in zip(sup, chunk) if r[2] == "answer")
        any_n += a / d
        pad_n += p / d
        pure += int(p / d > 0.9)
        den += 1
    return any_n / max(1, den), pad_n / max(1, den), pure / max(1, den)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_n", type=int, default=8000)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--format", default="moved2")
    ap.add_argument("--ratios", type=int, nargs="+", default=[0, 1, 4, 16])
    a = ap.parse_args()

    specs = E.three_cell_specs(P.TRAIN_LENGTHS)
    world, r = TK.build_world(specs["composed"])
    tok = Tokenizer.build([world], r)
    print(f"batch={a.batch}  train_n={a.train_n}  format={a.format}\n")
    print(f"{'stage':22s} {'ratio':>5s} {'sort':>10s} {'any_share':>10s} {'pad_share':>10s} "
          f"{'pad_pure':>9s}")
    for name, _share, weights in E.SCHEDULE:
        for ratio in a.ratios:
            for group in (False, True):
                rows = build(specs, weights, a.train_n, tok, a.format,
                             answer_ratio=ratio, group_masked=group)
                sh, ps, pp = answer_share(rows, a.batch)
                print(f"{name:22s} {ratio:5d} {('grouped' if group else 'length'):>10s} "
                      f"{sh:10.4f} {ps:10.4f} {pp:9.3f}")
    # the DENSE control, whose plain read forms on 3 of 3 seeds
    print()
    for name, _share, weights in E.SCHEDULE[:1]:
        rows = []
        for arm, share in sorted(weights.items()):
            n = int(round(a.train_n * share))
            for e in TK.generate(specs[arm], "train", n=n):
                rows.append((len(tok.encode(f"{e.prompt} {e.answer}", add_eos=True)),
                             len(tok.encode(e.prompt)), "plain"))
                if "interleaved_prompt" in e.meta:
                    rows.append((min(len(tok.encode(f"{e.meta['interleaved_prompt']} {e.answer}",
                                                    add_eos=True)), B.MAX_DOC_TOKENS), 1, "pad"))
        rows.sort(key=lambda x: x[0])
        sh, ps, pp = answer_share(rows, a.batch)
        print(f"{'dense (shipped)':22s} {'-':>5s} {'length':>10s} {sh:10.4f} {ps:10.4f} "
              f"{pp:9.3f}")


if __name__ == "__main__":
    main()

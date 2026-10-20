"""Diagnostics of a submission against the public window: per GT segment, what did we predict
and what did the evidence look like. Aggregates only.

    python analysis/diag.py out/SyntaxEror.csv ../mtah/evaluation/gt/BAR-ATH_possession_gt_public.csv
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from possession import model  # noqa: E402
from possession.features import load  # noqa: E402

sub = pd.read_csv(sys.argv[1])["label"].values
gt = pd.read_csv(sys.argv[2])["label"].values
f = load()
p = {a.dest: a.default for a in model.argparser()._actions if a.dest != "help"}
parts = model.evidence_parts(f, p)

idx = np.where(gt != "X")[0]
segs, s = [], idx[0]
for i in idx[1:]:
    if gt[i] != gt[s]:
        segs.append((s, i - 1)); s = i
segs.append((s, idx[-1]))

print("{:>6} {:>6} {:>5} {:>3}  {:<22} {}".format("start", "end", "len", "gt", "pred H/A/D", "  ".join("{:>6}".format(k) for k in parts)))
for a, b in segs:
    sl = slice(a, b + 1)
    pred = "/".join(str(int((sub[sl] == l).sum())) for l in "HAD")
    print("{:>6} {:>6} {:>5} {:>3}  {:<22} {}".format(a, b, b - a + 1, gt[a], pred,
          "  ".join("{:>6.2f}".format(np.nanmean(v[sl])) for v in parts.values())))

# transitions: for each GT change, the offset of the nearest predicted change of the same kind
gch = [i for i in idx[1:] if gt[i] != gt[i - 1]]
pch = np.array([i for i in idx[1:] if sub[i] != sub[i - 1]])
print("\nGT transitions and nearest predicted transition (frames, + = late):")
for i in gch:
    d = pch - i
    j = np.argmin(np.abs(d)) if len(d) else None
    print("  {:>6} {}->{}  pred {}->{} off {:+d}".format(i, gt[i - 1], gt[i], sub[pch[j] - 1], sub[pch[j]], int(d[j])))

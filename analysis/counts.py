"""Full-match segment counts of submissions: a plausibility check that owes nothing to the
public window. A real match has a few hundred possessions and about a hundred stoppages.

    python analysis/counts.py out/*.csv
"""
import os
import sys

import numpy as np
import pandas as pd


def segments(lab):
    idx = np.where(lab != "X")[0]
    out, s = [], idx[0]
    for prev, i in zip(idx[:-1], idx[1:]):
        if lab[i] != lab[s] or i != prev + 1:
            out.append((lab[s], prev - s + 1))
            s = i
    out.append((lab[s], idx[-1] - s + 1))
    return out


if __name__ == "__main__":
    for path in sys.argv[1:]:
        sg = segments(pd.read_csv(path)["label"].values)
        poss = [d for l, d in sg if l in "HA"]
        dead = [d for l, d in sg if l == "D"]
        print("{:<24} possessions {:>4} (mean {:>5.1f}s, <2s {:>3})   stoppages {:>4} (mean {:>5.1f}s)   D share {:.2f}".format(
            os.path.basename(path)[:-4], len(poss), np.mean(poss) / 25, sum(d < 50 for d in poss),
            len(dead), np.mean(dead) / 25, sum(dead) / (sum(dead) + sum(poss))))

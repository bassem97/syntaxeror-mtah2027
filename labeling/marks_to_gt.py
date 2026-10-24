"""Turn exported marks into a full-length frame,label ground-truth file the organisers' scorer reads.

A marks file has one row per state change: `frame,label`, the frame where that state starts.
Labels run until the next mark. Everything outside the given window is X, so the file scores only
the window, exactly like the public ground truth.

    python labeling/marks_to_gt.py marks_50000_57499.csv --window 50000-57499 -o labeling/data/gt_w1.csv
    python labeling/marks_to_gt.py a.csv b.csv c.csv --window 50000-57499 --window ... -o gt_internal.csv

Several marks files and windows can be merged into one file; the public ground truth can be
merged in too (`--public path`) so the internal set includes the organisers' five minutes.

Score any submission against it with the organisers' scorer:

    python ../mtah/evaluation/run_evaluation.py --gt labeling/data/gt_internal.csv --results out

The window should start on a frame you have marked (otherwise the frames before the first mark
are X) and the first mark inside a window should normally be the state in force at its start.
"""
import argparse
import csv
import os

import numpy as np

N_FRAMES = 199319
LABELS = set("HADX")


def read_marks(path):
    marks = {}
    with open(path, newline="") as fh:
        rows = list(csv.reader(fh))
    for row in rows[1:] if rows and rows[0][0].strip().lower() == "frame" else rows:
        if len(row) < 2:
            continue
        fr, label = int(row[0]), row[1].strip().upper()
        if label not in LABELS:
            raise ValueError("{}: bad label {!r} at frame {}".format(path, label, fr))
        marks[fr] = label
    return dict(sorted(marks.items()))


def expand(marks, window):
    """Labels over one window from the marks that fall in or before it."""
    a, b = window
    labels = np.full(b - a + 1, "X", dtype="<U1")
    current = "X"
    for fr, label in marks.items():
        if fr > b:
            break
        start = max(fr, a)
        if fr <= b:
            labels[start - a:] = label
            current = label
    return labels


def build(mark_paths, windows, public=None, n_frames=N_FRAMES):
    gt = np.full(n_frames, "X", dtype="<U1")
    if public:
        with open(public, newline="") as fh:
            for row in list(csv.reader(fh))[1:]:
                gt[int(row[0])] = row[1].strip().upper()
    for path in mark_paths:
        marks = read_marks(path)
        for a, b in windows:
            if any(a <= fr <= b for fr in marks):
                gt[a:b + 1] = expand(marks, (a, b))
    return gt


def write(gt, path):
    folder = os.path.dirname(path)
    if folder:
        os.makedirs(folder, exist_ok=True)
    with open(path, "w", newline="") as fh:
        fh.write("frame,label\n")
        fh.writelines("{},{}\n".format(i, l) for i, l in enumerate(gt))


def summary(gt, short_frames=10):
    graded = gt != "X"
    counts = {l: int((gt == l).sum()) for l in "HAD"}
    runs, short = 0, []
    start, prev = 0, "X"
    for i, l in enumerate(gt):
        if l != prev:
            if prev != "X" and i - start < short_frames:
                short.append("{} {} x{}".format(start, prev, i - start))
            if l in "HA":
                runs += 1
            start, prev = i, l
    text = "{} graded frames ({:.1f} min), H {H} A {A} D {D}, {} possessions".format(
        graded.sum(), graded.sum() / 25 / 60, runs, **counts)
    if short:
        text += "\n  states shorter than {} frames, probably double presses -- review: {}".format(
            short_frames, ", ".join(short))
    return text


def self_check():
    marks = {100: "H", 150: "A", 160: "D", 300: "H"}
    labels = expand(marks, (120, 200))
    assert "".join(labels[:35]) == "H" * 30 + "A" * 5, labels[:35]
    assert labels[40] == "D" and labels[-1] == "D"
    assert "".join(expand({50: "A"}, (100, 102))) == "AAA"
    assert "".join(expand({}, (0, 2))) == "XXX"


if __name__ == "__main__":
    self_check()
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("marks", nargs="+", help="exported marks CSV files")
    ap.add_argument("--window", action="append", required=True, help="first-last frame, repeatable")
    ap.add_argument("--public", help="public ground truth CSV to merge in")
    ap.add_argument("-o", "--out", required=True)
    ap.add_argument("--n-frames", type=int, default=N_FRAMES, help="length of the output (another match's frame count)")
    args = ap.parse_args()
    windows = [tuple(int(x) for x in w.split("-")) for w in args.window]
    gt = build(args.marks, windows, args.public, args.n_frames)
    write(gt, args.out)
    print("wrote", args.out, "->", summary(gt))

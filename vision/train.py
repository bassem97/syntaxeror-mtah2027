"""Train the ball detector on the crops from vision/dataset.py.

    python vision/train.py --epochs 10            # -> data/cache/ballnet.pt
    python vision/train.py --epochs 1 --limit 500 # smoke test

Holds out every 10th frame (by frame index, so neighbouring frames never straddle the split) and
reports, on the held-out positives, how often the heatmap peak lands within 8 px of the label and
the score separation between crops with and without a ball.
"""
import argparse
import csv
import os
import sys
import time

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from possession.games import DATA_DIR  # noqa: E402
from vision.model import STRIDE, BallNet, focal_loss, gaussian_target, peaks  # noqa: E402

CROPS = os.path.join(DATA_DIR, "cache", "crops")
WEIGHTS = os.path.join(DATA_DIR, "cache", "ballnet.pt")


class Crops(Dataset):
    def __init__(self, rows, augment):
        self.rows = rows
        self.augment = augment

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        name, frame, u, v, iso = self.rows[i]
        img = cv2.imread(os.path.join(CROPS, name))
        u, v = float(u), float(v)
        if self.augment:
            if np.random.rand() < 0.5:  # horizontal flip
                img = img[:, ::-1]
                if u >= 0:
                    u = img.shape[1] - 1 - u
            # brightness / contrast jitter
            a = np.random.uniform(0.8, 1.2)
            b = np.random.uniform(-20, 20)
            img = np.clip(img.astype(np.float32) * a + b, 0, 255).astype(np.uint8)
        x = torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1)
        c = torch.tensor([u / STRIDE, v / STRIDE] if u >= 0 else [-1.0, -1.0])
        return x, c, iso


def read_rows():
    with open(os.path.join(CROPS, "labels.csv"), newline="") as fh:
        rows = [(r["file"], int(r["frame"]), float(r["u"]), float(r["v"]), int(r.get("iso", 0))) for r in csv.DictReader(fh)]
    train = [r for r in rows if (r[1] // 8) % 10 != 0]
    val = [r for r in rows if (r[1] // 8) % 10 == 0]
    return train, val


def evaluate(model, loader, device):
    model.eval()
    hits, n_pos, pos_scores, neg_scores = 0, 0, [], []
    iso_hits, iso_n = 0, 0
    with torch.no_grad():
        for x, c, iso in loader:
            x = x.to(device)
            u, v, s = peaks(model(x))
            for k in range(len(c)):
                if c[k, 0] >= 0:
                    n_pos += 1
                    d = np.hypot(u[k].item() - c[k, 0].item() * STRIDE, v[k].item() - c[k, 1].item() * STRIDE)
                    hits += d <= 8
                    if iso[k]:
                        iso_n += 1
                        iso_hits += d <= 8
                    pos_scores.append(s[k].item())
                else:
                    neg_scores.append(s[k].item())
    model.train()
    pos_scores, neg_scores = np.array(pos_scores), np.array(neg_scores)
    thr = 0.5
    return {"hit8": hits / max(n_pos, 1), "hit8_iso": iso_hits / max(iso_n, 1), "pos_score_med": float(np.median(pos_scores)) if len(pos_scores) else 0,
            "neg_score_p95": float(np.percentile(neg_scores, 95)) if len(neg_scores) else 0,
            "recall@0.5": float((pos_scores > thr).mean()) if len(pos_scores) else 0,
            "false_alarm@0.5": float((neg_scores > thr).mean()) if len(neg_scores) else 0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--limit", type=int, help="use only the first N training crops (smoke test)")
    ap.add_argument("--out", default=WEIGHTS)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train, val = read_rows()
    if args.limit:
        train, val = train[:args.limit], val[:max(50, args.limit // 10)]
    print("device {}  train {} crops  val {} crops".format(device, len(train), len(val)))
    tl = DataLoader(Crops(train, True), batch_size=args.batch, shuffle=True, num_workers=4, drop_last=True, persistent_workers=True)
    vl = DataLoader(Crops(val, False), batch_size=args.batch, num_workers=2)
    model = BallNet().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.epochs * len(tl))
    for epoch in range(args.epochs):
        t0, total, steps = time.time(), 0.0, 0
        for x, c, _ in tl:
            x, c = x.to(device), c.to(device)
            logits = model(x)
            target = gaussian_target(len(x), logits.shape[-2], logits.shape[-1], c, device=device)
            loss = focal_loss(logits, target)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            total += loss.item()
            steps += 1
        stats = evaluate(model, vl, device)
        print("epoch {:>2}  loss {:.3f}  {:.0f}s  val: peak within 8px {:.3f} (isolated {:.3f})  recall@0.5 {:.3f}  false alarm@0.5 {:.3f}  pos score {:.2f}  neg p95 {:.2f}".format(
            epoch + 1, total / max(steps, 1), time.time() - t0, stats["hit8"], stats["hit8_iso"], stats["recall@0.5"], stats["false_alarm@0.5"],
            stats["pos_score_med"], stats["neg_score_p95"]), flush=True)
        torch.save(model.state_dict(), args.out)
    print("saved", args.out)


if __name__ == "__main__":
    main()

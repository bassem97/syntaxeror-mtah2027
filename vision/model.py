"""A small fully convolutional ball detector: image in, heatmap of "ball centre here" out.

CenterNet-style, stride 4: a 1920x1080 frame gives a 480x270 heatmap whose peak is the ball and
whose peak value is the confidence. Fully convolutional, so it trains on 256 px crops and runs on
whole frames. Plain PyTorch, about 300k parameters: the ball is a bright blob of 6-14 px on grass
seen from a fixed camera, which does not need a big network.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

STRIDE = 4


def block(cin, cout, stride=1):
    return nn.Sequential(nn.Conv2d(cin, cout, 3, stride, 1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


class BallNet(nn.Module):
    def __init__(self, width=32):
        super().__init__()
        w = width
        self.enc1 = nn.Sequential(block(3, w), block(w, w))                 # full res
        self.enc2 = nn.Sequential(block(w, 2 * w, 2), block(2 * w, 2 * w))  # /2
        self.enc3 = nn.Sequential(block(2 * w, 4 * w, 2), block(4 * w, 4 * w))  # /4
        self.enc4 = nn.Sequential(block(4 * w, 4 * w, 2), block(4 * w, 4 * w))  # /8, context
        self.dec3 = block(8 * w, 4 * w)                                     # /4 after upsample + skip
        self.head = nn.Conv2d(4 * w, 1, 1)
        nn.init.constant_(self.head.bias, -4.0)  # start with "no ball anywhere"

    def forward(self, x):
        x = x.float() / 255.0
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        up = F.interpolate(e4, size=e3.shape[-2:], mode="bilinear", align_corners=False)
        d3 = self.dec3(torch.cat([e3, up], 1))
        return self.head(d3)  # logits, (n, 1, H/4, W/4)


def gaussian_target(n, h, w, centres, sigma=1.5, device="cpu"):
    """Heatmap targets at stride resolution; centres in stride units, (-1, -1) = no ball."""
    ys = torch.arange(h, device=device).view(1, h, 1).float()
    xs = torch.arange(w, device=device).view(1, 1, w).float()
    # centres snap to a grid cell so every ball has exactly one pixel at 1.0: the focal loss
    # treats only those as positives, and a fractional centre would leave a crop with none
    cx = torch.round(centres[:, 0]).clamp(0, w - 1).view(n, 1, 1)
    cy = torch.round(centres[:, 1]).clamp(0, h - 1).view(n, 1, 1)
    g = torch.exp(-((xs - cx) ** 2 + (ys - cy) ** 2) / (2 * sigma ** 2))
    g[(centres[:, 0] < 0)] = 0.0
    return g.unsqueeze(1)


def focal_loss(logits, target, alpha=2.0, beta=4.0):
    """CenterNet focal loss on a Gaussian target. Positives are the peak pixels (target == 1)."""
    p = torch.sigmoid(logits).clamp(1e-4, 1 - 1e-4)
    pos = (target >= 0.999).float()
    neg_w = (1 - target) ** beta
    pos_loss = -(torch.log(p) * (1 - p) ** alpha * pos).sum()
    neg_loss = -(torch.log(1 - p) * p ** alpha * neg_w * (1 - pos)).sum()
    n_pos = pos.sum().clamp(min=1.0)
    return (pos_loss + neg_loss) / n_pos


def peaks(logits):
    """(u, v, score) of the heatmap maximum per image, in input pixels."""
    u, v, s = topk_peaks(logits, 1)
    return u[:, 0], v[:, 0], s[:, 0]


def topk_peaks(logits, k=5):
    """The k strongest local maxima per image, (n, k) each, in input pixels. A 3x3 max-pool
    keeps one entry per blob, so the k peaks are k different places, not one blob k times."""
    prob = torch.sigmoid(logits)
    n, _, h, w = prob.shape
    pooled = F.max_pool2d(prob, 3, stride=1, padding=1)
    prob = prob * (pooled == prob)
    flat = prob.view(n, -1)
    score, idx = flat.topk(k, dim=1)
    v = (idx // w).float() * STRIDE + STRIDE / 2
    u = (idx % w).float() * STRIDE + STRIDE / 2
    return u, v, score

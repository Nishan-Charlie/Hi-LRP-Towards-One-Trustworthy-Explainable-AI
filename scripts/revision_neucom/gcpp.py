"""R3.4: measure the quantity the Grad-CAM++ instability argument rests on.

The benchmark's Grad-CAM++ (like pytorch-grad-cam and Eq. 19 of Chattopadhay et
al.) uses the closed form obtained under the exponential-score assumption, in
which the 2nd/3rd derivatives are replaced by powers of the first-order gradient:
    alpha_ij^k = g^2 / (2 g^2 + S_k g^3) = 1 / (2 + S_k g_ij)   (g != 0),
    S_k = sum_ab A^k_ab.
alpha has a pole at S_k g_ij = -2. Per image we record, over all (k, i, j) with
g_ij^k > 0 (the only entries that enter the weights):
  core_min      min |2 + S_k g|
  frac_pole     share of entries with |2 + S_k g| < 0.5  (|alpha| > 2)
  alpha_max     max |alpha|
  sg_p99        99th percentile of |S_k g|
  g_cv          mean over channels of std_ij(g)/|mean_ij(g)| (0 when the logit is
                linear in the global average of the map, i.e. GAP + linear head)
  neg_S         share of channels with S_k < 0
  cam_zero      1 if the Grad-CAM++ map is identically zero (degenerate)

usage: python gcpp.py MODEL N
"""
import csv
import os
import sys

import numpy as np
import torch

from common import OUT, DEVICE, load_data, load_model, cam_map

model_name, N = sys.argv[1], int(sys.argv[2])
data = load_data(N)
mw = load_model(model_name)
rows = []
for d in data:
    x = d["image"].unsqueeze(0).to(DEVICE)
    cam, parts = cam_map(mw, x, d["label"], plus_plus=True, return_parts=True)
    G, S = parts["G"].double().cpu(), parts["S"].double().cpu()   # statistics on CPU
    cam = cam.cpu()
    core = (2 + S * G)
    pos = G > 0
    c = core[pos].abs()
    sg = (S * G)[pos].abs()
    gm = G.mean(dim=(2, 3)).abs().squeeze(0)
    gs = G.std(dim=(2, 3)).squeeze(0)
    valid = gm > 1e-12
    rows.append(dict(model=model_name, idx=d["idx"],
                     core_min=float(c.min()) if c.numel() else np.nan,
                     frac_pole=float((c < 0.5).double().mean()) if c.numel() else np.nan,
                     alpha_max=float((1 / c).max()) if c.numel() else np.nan,
                     sg_p99=float(torch.quantile(sg.float(), 0.99)) if sg.numel() else np.nan,
                     g_cv=float((gs[valid] / gm[valid]).mean()) if valid.any() else np.nan,
                     neg_S=float((S.squeeze() < 0).double().mean()),
                     cam_zero=int(float(cam.abs().max()) == 0.0)))
os.makedirs(OUT, exist_ok=True)
out = os.path.join(OUT, f"gcpp_{model_name}_n{N}.csv")
with open(out, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)
import pandas as pd
df = pd.DataFrame(rows)
print(model_name, df.drop(columns=["model", "idx"]).median().round(4).to_dict(), "cam_zero_rate", df.cam_zero.mean())

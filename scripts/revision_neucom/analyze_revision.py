"""Aggregate every NEUCOM revision experiment into tables, statistics and figures.

Reads results/revision_neucom/*.csv (written by acc.py, loc.py, ms.py, gcpp.py,
ffe.py) and results/revision_vit_native/summary.csv (runner), and writes
results/revision_neucom/analysis/{summary.json, *.tex} and figures/revision/*.pdf.
Safe to re-run on partial results; missing pieces are skipped.
"""
import glob
import itertools
import json
import os
import warnings

import numpy as np
import pandas as pd
from scipy import stats

import common  # noqa: F401  (cwd -> repo root)
from common import MAIN_MODELS, SHORT

warnings.filterwarnings("ignore")
R = os.path.join("results", "revision_neucom")
A = os.path.join(R, "analysis")
FIG = os.path.join("figures", "revision")
os.makedirs(A, exist_ok=True)
os.makedirs(FIG, exist_ok=True)
S = {}

METHODS = ["saliency", "integrated_gradients", "input_x_gradient", "smoothgrad", "vargrad", "gradient_shap",
           "grad_cam", "grad_cam_plus_plus", "attention_rollout", "attnlrp", "occlusion", "rise", "lime"]
MLAB = {"saliency": "Saliency", "integrated_gradients": "Integrated Grad", "input_x_gradient": "Input$\\times$Grad",
        "smoothgrad": "SmoothGrad", "vargrad": "VarGrad", "gradient_shap": "GradientSHAP",
        "grad_cam": "Grad-CAM", "grad_cam_plus_plus": "Grad-CAM++", "grad_cam_plus_plus_stab": "Grad-CAM++ (bounded $\\alpha$)",
        "attention_rollout": "Attention Rollout", "attnlrp": "AttnLRP", "occlusion": "Occlusion", "rise": "RISE", "lime": "LIME"}
COMPLETE = [m for m in METHODS if m not in ("attention_rollout", "attnlrp")]
SIGNED = ["integrated_gradients", "input_x_gradient", "gradient_shap", "occlusion", "lime", "attnlrp"]

FAMILY = {"resnet18": "CNN", "resnet50": "CNN", "resnet101": "CNN", "convnext_tiny": "CNN", "convnext_base": "CNN",
          "vit_small_patch16_224": "Isotropic", "vit_base_patch16_224": "Isotropic",
          "deit_small_patch16_224": "Isotropic", "deit_base_patch16_224": "Isotropic",
          "swin_tiny_patch4_window7_224": "Hierarchical", "swin_base_patch4_window7_224": "Hierarchical",
          "pvt_v2_b0": "Hierarchical", "pvt_v2_b2": "Hierarchical", "pvt_v2_b4": "Hierarchical",
          "maxvit_tiny_tf_224": "Multi-axis", "maxvit_small_tf_224": "Multi-axis",
          "mobilevitv2_100": "Hybrid", "mobilevitv2_200": "Hybrid",
          "efficientvit_b1": "Linear", "efficientvit_b2": "Linear", "efficientvit_b3": "Linear"}
NAME = dict(SHORT, **{"resnet18": "ResNet-18", "resnet101": "ResNet-101", "convnext_tiny": "ConvNeXt-T",
                      "convnext_base": "ConvNeXt-B", "vit_small_patch16_224": "ViT-S/16",
                      "deit_small_patch16_224": "DeiT-S", "deit_base_patch16_224": "DeiT-B",
                      "swin_tiny_patch4_window7_224": "Swin-T", "pvt_v2_b0": "PVT-v2-B0", "pvt_v2_b4": "PVT-v2-B4",
                      "maxvit_tiny_tf_224": "MaxViT-T", "mobilevitv2_200": "MobileViT-v2-2.0",
                      "efficientvit_b3": "EffViT-B3"})
# parameter counts (M) of the additional timm backbones (sum of p.numel())
EXTRA_PARAMS = {"resnet18": 11.7, "resnet101": 44.5, "convnext_tiny": 28.6, "convnext_base": 88.6,
                "vit_small_patch16_224": 22.1, "deit_small_patch16_224": 22.1, "deit_base_patch16_224": 86.6,
                "swin_tiny_patch4_window7_224": 28.3, "pvt_v2_b0": 3.7, "pvt_v2_b4": 62.6,
                "maxvit_tiny_tf_224": 30.9, "mobilevitv2_200": 18.4, "efficientvit_b3": 48.6}
NAME["resnet50"] = "ResNet-50"
NAME["mobilevitv2_100"] = "MobileViT-v2-1.0"
NAME["pvt_v2_b2"] = "PVT-v2-B2"
# how the class logit reads the CAM target map (determines the Grad-CAM++ pole, Section 8.2)
HEAD = {"resnet18": "GAP+linear", "resnet50": "GAP+linear", "resnet101": "GAP+linear",
        "swin_tiny_patch4_window7_224": "GAP+linear", "swin_base_patch4_window7_224": "GAP+linear",
        "pvt_v2_b0": "GAP+linear", "pvt_v2_b2": "GAP+linear", "pvt_v2_b4": "GAP+linear",
        "mobilevitv2_100": "GAP+linear", "mobilevitv2_200": "GAP+linear",
        "convnext_tiny": "GAP+LN+linear", "convnext_base": "GAP+LN+linear",
        "maxvit_tiny_tf_224": "GAP+LN+MLP", "maxvit_small_tf_224": "GAP+LN+MLP",
        "efficientvit_b1": "conv+GAP+LN-MLP", "efficientvit_b2": "conv+GAP+LN-MLP", "efficientvit_b3": "conv+GAP+LN-MLP",
        "vit_small_patch16_224": "CLS attention", "vit_base_patch16_224": "CLS attention",
        "deit_small_patch16_224": "CLS attention", "deit_base_patch16_224": "CLS attention"}


def f2(x, d=2):
    return "--" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{d}f}"


def save_tex(name, s):
    open(os.path.join(A, name), "w", encoding="utf-8").write(s)


# ------------------------------------------------------------------ accuracy
acc = pd.concat([pd.read_csv(f) for f in glob.glob(os.path.join(R, "accuracy_summary*.csv"))]).drop_duplicates(["model", "subset"])
acc_i = pd.concat([pd.read_csv(f) for f in glob.glob(os.path.join(R, "accuracy_per_image*.csv"))]).drop_duplicates(["model", "idx"])
S["accuracy"] = acc.to_dict("records")
rows = []
for m in MAIN_MODELS:
    a = acc[(acc.model == m) & (acc.subset == "n1000")].iloc[0]
    g = acc[(acc.model == m) & (acc.subset == "grid")].iloc[0]
    rows.append(f"{NAME[m]} & {a.params_M:.1f} & {100 * a.top1:.1f} & {100 * a.top5:.1f} & {a.conf_pred:.2f} & "
                f"{a.conf_pred_correct:.2f} & {a.conf_pred_wrong:.2f} & {100 * g.top1:.0f} ({int(g.n)}) \\\\")
save_tex("tab_accuracy.tex", "\n".join(rows))

# ------------------------------------------------------------------ localisation
loc_files = [f for f in glob.glob(os.path.join(R, "loc_*.csv"))]
if loc_files:
    L = pd.concat([pd.read_csv(f) for f in loc_files], ignore_index=True)
    L = L.drop_duplicates(["model", "method", "idx", "target_mode"], keep="last")
    gt = L[L.target_mode == "gt"].set_index(["model", "method", "idx"])
    pr = L[L.target_mode == "pred"].set_index(["model", "method", "idx"])
    # 'pred' protocol: predicted-class row when misclassified, otherwise the identical gt row
    P = gt.copy()
    common_idx = pr.index.intersection(gt.index)
    P.loc[common_idx, pr.columns.intersection(P.columns)] = pr.loc[common_idx, pr.columns.intersection(P.columns)]
    gt, P = gt.reset_index(), P.reset_index()

    def cell(df, model, method, col, sel=None):
        d = df[(df.model == model) & (df.method == method)]
        if sel is not None:
            d = d[sel(d)]
        v = d[col].dropna()
        return (float(v.mean()) if len(v) else np.nan), len(v)

    # main-backbone EBPG / bbox-PG tables (gt target, abs protocol)
    loc_tab = {}
    for col in ("ebpg_abs", "bboxpg_abs", "densepg_abs"):
        loc_tab[col] = {me: {mo: cell(gt, mo, me, col) for mo in MAIN_MODELS} for me in METHODS}
    S["loc_gt_abs"] = {c: {me: {mo: v[0] for mo, v in d.items()} for me, d in t.items()} for c, t in loc_tab.items()}
    S["loc_n"] = {me: {mo: v[1] for mo, v in d.items()} for me, d in loc_tab["ebpg_abs"].items()}
    masks = common.load_data(1000)
    S["mask_area"] = {"n1000": float(np.mean([float(d["mask"].mean()) for d in masks])),
                      "n100": float(np.mean([float(d["mask"].mean()) for d in masks[:100]]))}
    lines = []
    for me in METHODS:
        vals = [loc_tab["ebpg_abs"][me][mo][0] for mo in MAIN_MODELS]
        if all(not np.isfinite(v) for v in vals):
            continue
        best = []
        for j, mo in enumerate(MAIN_MODELS):
            col = [loc_tab["ebpg_abs"][x][mo][0] for x in METHODS]
            col = [c for c in col if np.isfinite(c)]
            # bold every cell that ties the column maximum at the printed precision
            best.append(np.isfinite(vals[j]) and f2(vals[j]) == f2(max(col)))
        dag = "$^{\\dagger}$" if me in ("integrated_gradients", "occlusion", "rise", "lime") else ""
        cells = [("\\best{" + f2(v) + "}" if b else f2(v)) if np.isfinite(v) else "\\nd" for v, b in zip(vals, best)]
        lines.append(f"{MLAB[me]}{dag} & " + " & ".join(cells) + " \\\\")
    save_tex("tab_ebpg_all.tex", "\n".join(lines))

    # ---------------- target protocol (R1.2 / R3.2)
    tp = {}
    for mo in MAIN_MODELS:
        for me in METHODS:
            g_all = cell(gt, mo, me, "ebpg_abs")
            if not np.isfinite(g_all[0]):
                continue
            tp[(mo, me)] = dict(
                ebpg_gt=g_all[0], n=g_all[1],
                ebpg_pred=cell(P, mo, me, "ebpg_abs")[0],
                ebpg_correct=cell(gt, mo, me, "ebpg_abs", lambda d: d.correct == 1)[0],
                ebpg_wrong_gt=cell(gt, mo, me, "ebpg_abs", lambda d: d.correct == 0)[0],
                ebpg_wrong_pred=cell(P, mo, me, "ebpg_abs", lambda d: d.correct == 0)[0],
                pg_gt=cell(gt, mo, me, "bboxpg_abs")[0], pg_pred=cell(P, mo, me, "bboxpg_abs")[0],
                pg_correct=cell(gt, mo, me, "bboxpg_abs", lambda d: d.correct == 1)[0],
                n_correct=cell(gt, mo, me, "ebpg_abs", lambda d: d.correct == 1)[1])
    S["target_protocol"] = {f"{k[0]}|{k[1]}": v for k, v in tp.items()}
    lines, agree = [], {}
    for mo in MAIN_MODELS:
        ks = [k for k in tp if k[0] == mo]
        if not ks:
            continue
        mean = lambda key: np.nanmean([tp[k][key] for k in ks])
        rk = {}
        for key in ("ebpg_gt", "ebpg_pred", "ebpg_correct", "pg_gt", "pg_pred", "pg_correct"):
            rk[key] = [tp[k][key] for k in ks]
        rho_p = stats.spearmanr(rk["ebpg_gt"], rk["ebpg_pred"])[0]
        rho_c = stats.spearmanr(rk["ebpg_gt"], rk["ebpg_correct"])[0]
        rho_pg = stats.spearmanr(rk["pg_gt"], rk["pg_correct"])[0]
        maxdiff = np.nanmax(np.abs(np.array(rk["ebpg_gt"]) - np.array(rk["ebpg_correct"])))
        agree[mo] = dict(rho_pred=rho_p, rho_correct=rho_c, rho_pg_correct=rho_pg, maxdiff=maxdiff, k=len(ks))
        nc = acc[(acc.model == mo) & (acc.subset == "n1000")].iloc[0]
        lines.append(f"{NAME[mo]} & {mean('ebpg_gt'):.3f} & {mean('ebpg_pred'):.3f} & {mean('ebpg_correct'):.3f} & "
                     f"{mean('ebpg_wrong_gt'):.3f} & {mean('ebpg_wrong_pred'):.3f} & {rho_p:.2f} & {rho_c:.2f} & {maxdiff:.3f} \\\\")
    S["target_agreement"] = agree
    save_tex("tab_target.tex", "\n".join(lines))

    # ranking transfer across backbones under each protocol (EBPG, methods defined everywhere)
    def transfer(key):
        M = np.array([[tp.get((mo, me), {}).get(key, np.nan) for mo in MAIN_MODELS] for me in COMPLETE])
        ok = ~np.isnan(M).any(axis=1)
        M = M[ok]
        if M.shape[0] < 4:
            return None
        rs = [stats.spearmanr(M[:, a], M[:, b])[0] for a, b in itertools.combinations(range(M.shape[1]), 2)]
        return dict(mean=float(np.mean(rs)), cnn_effb2=float(stats.spearmanr(M[:, 0], M[:, -1])[0]), k=int(M.shape[0]))
    S["transfer_ebpg"] = {k: transfer(k) for k in ("ebpg_gt", "ebpg_pred", "ebpg_correct", "pg_gt", "pg_correct")}

    # ---------------- normalisation protocol (R1.4)
    norm = {}
    for me in SIGNED:
        for mo in MAIN_MODELS:
            if not np.isfinite(cell(gt, mo, me, "ebpg_abs")[0]):
                continue
            norm[(me, mo)] = {f"{c}_{p}": cell(gt, mo, me, f"{c}_{p}")[0]
                              for c in ("ebpg", "bboxpg") for p in ("abs", "pos", "signed")}
    ffe = [pd.read_csv(f) for f in glob.glob(os.path.join(R, "ffe_*.csv"))]
    if ffe:
        F = pd.concat(ffe)
        for (me, mo, p), d in F.groupby(["method", "model", "protocol"]):
            if (me, mo) in norm:
                norm[(me, mo)][f"fc_{p}"] = float(d.fc.mean())
                norm[(me, mo)][f"fe_{p}"] = float(d.fe.mean())
    S["normalisation"] = {f"{k[0]}|{k[1]}": v for k, v in norm.items()}
    lines = []
    for me in SIGNED:
        ks = [k for k in norm if k[0] == me]
        if not ks:
            continue
        av = lambda key: np.nanmean([norm[k].get(key, np.nan) for k in ks])
        lines.append(f"{MLAB[me]} ({len(ks)}) & " + " & ".join(
            f2(av(f"{c}_{p}"), 3 if c in ('fc', 'fe') else 2) for c in ("bboxpg", "ebpg", "fc", "fe") for p in ("abs", "pos", "signed")) + " \\\\")
    save_tex("tab_normalisation.tex", "\n".join(lines))
    # does the protocol change method rankings? (all 11 complete methods, EBPG and PG, per backbone)
    rk = {}
    for mo in MAIN_MODELS:
        for c in ("ebpg", "bboxpg"):
            a = [cell(gt, mo, me, f"{c}_abs")[0] for me in COMPLETE]
            p = [cell(gt, mo, me, f"{c}_pos")[0] for me in COMPLETE]
            ok = np.isfinite(a) & np.isfinite(p)
            if ok.sum() >= 4:
                rk[f"{mo}|{c}"] = float(stats.spearmanr(np.array(a)[ok], np.array(p)[ok])[0])
    S["normalisation_rank_rho"] = rk

# ------------------------------------------------------------------ Max-Sensitivity, multi-seed
ms_files = glob.glob(os.path.join(R, "ms_*.csv"))
if ms_files:
    MS = pd.concat([pd.read_csv(f) for f in ms_files], ignore_index=True)
    MS = MS.drop_duplicates(["model", "method", "seed", "idx"], keep="last")
    scols = [c for c in MS.columns if c.startswith("s") and c[1:].isdigit()]
    MS["ms3"] = MS[["s0", "s1", "s2"]].max(axis=1, skipna=False)
    full = [c for c in scols if MS[c].notna().any()]
    MS["k"] = MS[scols].notna().sum(axis=1)
    MS["msK"] = MS[scols].max(axis=1)
    out = {}
    for (mo, me), d in MS.groupby(["model", "method"]):
        per_seed3 = d.groupby("seed").ms3.mean()
        per_seedK = d.groupby("seed").msK.mean()
        out[f"{mo}|{me}"] = dict(n_img=int(d.idx.nunique()), seeds=int(d.seed.nunique()), K=int(d.k.max()),
                                 mean3=float(d.ms3.mean()), med3=float(d.ms3.median()),
                                 meanK=float(d.msK.mean()), medK=float(d.msK.median()),
                                 p90K=float(d.msK.quantile(0.9)), maxK=float(d.msK.max()),
                                 seed_mean3_min=float(per_seed3.min()), seed_mean3_max=float(per_seed3.max()),
                                 seed_meanK_min=float(per_seedK.min()), seed_meanK_max=float(per_seedK.max()),
                                 seed_med3=[float(x) for x in d.groupby("seed").ms3.median()],
                                 degenerate=float(d.degenerate.mean()), nan=float(d.msK.isna().mean()))
    S["ms"] = out
    # rank stability of methods across seeds (mean MS@3, the original statistic, and median MS@K)
    stab = {}
    for mo in MS.model.unique():
        for stat_, col in (("mean3", "ms3"), ("medK", "msK")):
            piv = MS[MS.model == mo].groupby(["method", "seed"])[col].agg("mean" if stat_ == "mean3" else "median").unstack()
            piv = piv.dropna()
            if piv.shape[0] >= 4 and piv.shape[1] >= 2:
                rs = [stats.spearmanr(piv[a], piv[b])[0] for a, b in itertools.combinations(piv.columns, 2)]
                stab[f"{mo}|{stat_}"] = float(np.mean(rs))
    S["ms_seed_rank_stability"] = stab
    # table: median MS@10 [seed range of the mean MS@3] for main backbones
    lines = []
    for me in ["saliency", "input_x_gradient", "smoothgrad", "vargrad", "gradient_shap", "grad_cam",
               "grad_cam_plus_plus", "grad_cam_plus_plus_stab", "attention_rollout", "attnlrp",
               "integrated_gradients", "rise"]:
        cells = []
        for mo in MAIN_MODELS:
            o = out.get(f"{mo}|{me}")
            if o is None:
                cells.append("\\nd")
            else:
                cells.append(f"{f2(o['medK'])} \\scriptsize{{({f2(o['seed_mean3_min'], 1)}--{f2(o['seed_mean3_max'], 1)})}}")
        if all(c == "\\nd" for c in cells):
            continue
        lines.append(f"{MLAB[me]} & " + " & ".join(cells) + " \\\\")
    save_tex("tab_ms_seeds.tex", "\n".join(lines))

# ------------------------------------------------------------------ Grad-CAM++ mechanism
gc_files = glob.glob(os.path.join(R, "gcpp_*.csv"))
if gc_files:
    G = pd.concat([pd.read_csv(f) for f in gc_files], ignore_index=True).drop_duplicates(["model", "idx"])
    mech = {}
    for mo, d in G.groupby("model"):
        o = dict(n=len(d), g_cv=float(d.g_cv.median()), core_min=float(d.core_min.median()),
                 near_pole=float((d.core_min < 0.5).mean()), alpha_max=float(d.alpha_max.median()),
                 neg_S=float(d.neg_S.median()), cam_zero=float(d.cam_zero.mean()), head=HEAD.get(mo, "?"),
                 family=FAMILY.get(mo, "?"))
        if ms_files:
            for me in ("grad_cam", "grad_cam_plus_plus", "grad_cam_plus_plus_stab"):
                e = S["ms"].get(f"{mo}|{me}")
                if e:
                    o[f"ms_{me}_med"], o[f"ms_{me}_mean"], o[f"ms_{me}_p90"] = e["medK"], e["meanK"], e["p90K"]
        mech[mo] = o
    S["mechanism"] = mech
    # per-image association: proximity to the pole vs Grad-CAM++ sensitivity
    if ms_files:
        gpp = MS[MS.method == "grad_cam_plus_plus"].groupby(["model", "idx"]).msK.mean().reset_index()
        J = gpp.merge(G, on=["model", "idx"])
        if len(J) > 20:
            rho_all = stats.spearmanr(J.core_min, J.msK, nan_policy="omit")[0]
            within = {mo: float(stats.spearmanr(d.core_min, d.msK, nan_policy="omit")[0])
                      for mo, d in J.groupby("model") if len(d) > 20 and d.core_min.std() > 0}
            near = J[J.core_min < 0.5].msK.dropna()  # NaN = all-zero map, MS undefined
            far = J[J.core_min >= 0.5].msK.dropna()
            S["mechanism_assoc"] = dict(rho_pooled=float(rho_all), rho_within=within, n=len(J),
                                        med_ms_near=float(near.median()) if len(near) else None,
                                        med_ms_far=float(far.median()) if len(far) else None,
                                        n_near=int(len(near)), n_far=int(len(far)),
                                        mw_p=float(stats.mannwhitneyu(near, far).pvalue) if len(near) > 5 and len(far) > 5 else None)
    order = sorted(mech, key=lambda m: (list(HEAD.values()).index(HEAD.get(m, "GAP+linear")) if m in HEAD else 99, m))
    lines = []
    for mo in order:
        o = mech[mo]
        lines.append(f"{NAME.get(mo, mo)} & {o['head']} & {f2(o['g_cv'], 1)} & {f2(o['core_min'])} & "
                     f"{100 * o['near_pole']:.0f} & {f2(o['alpha_max'], 1)} & {100 * o['cam_zero']:.1f} & "
                     f"{f2(o.get('ms_grad_cam_med'))} & {f2(o.get('ms_grad_cam_plus_plus_med'))} "
                     f"({f2(o.get('ms_grad_cam_plus_plus_mean'), 1)}) & {f2(o.get('ms_grad_cam_plus_plus_stab_med'))} \\\\")
    save_tex("tab_mechanism.tex", "\n".join(lines))

# ------------------------------------------------------------------ capacity vs architecture
if loc_files:
    cap = {}
    for mo in FAMILY:
        a = acc[(acc.model == mo) & (acc.subset == "n1000")]
        if not a.empty:
            params, top1 = float(a.params_M.iloc[0]), float(a.top1.iloc[0])
        else:  # additional backbones: accuracy from the localization runs (all 1,000 images)
            d = gt[(gt.model == mo) & (gt.method == "grad_cam")]
            if d.empty or mo not in EXTRA_PARAMS:
                continue
            params, top1 = EXTRA_PARAMS[mo], float(d.correct.mean())
        o = dict(family=FAMILY[mo], params=params, top1=top1)
        for me in ("grad_cam", "grad_cam_plus_plus", "smoothgrad"):
            d = gt[(gt.model == mo) & (gt.method == me)]
            o[f"ebpg_{me}"] = float(d.ebpg_abs.mean()) if len(d) else np.nan
            o[f"pg_{me}"] = float(d.bboxpg_abs.mean()) if len(d) else np.nan
            o[f"n_{me}"] = int(len(d))
        if ms_files:
            for me in ("grad_cam", "grad_cam_plus_plus"):
                e = S["ms"].get(f"{mo}|{me}")
                o[f"ms_{me}_med"] = e["medK"] if e else np.nan
        cap[mo] = o
    S["capacity"] = cap
    C = pd.DataFrame(cap).T
    C["logp"] = np.log10(C.params.astype(float))
    anova = {}
    for y in [c for c in C.columns if c.startswith(("ebpg_", "pg_", "ms_"))]:
        d = C[["family", "logp", y]].dropna()
        d = d[d.family != "?"]
        if len(d) < 8:
            continue
        yv = d[y].astype(float).values
        Xf = pd.get_dummies(d.family, drop_first=True).astype(float).values
        X_full = np.column_stack([np.ones(len(d)), Xf, d.logp.astype(float).values])
        X_fam = np.column_stack([np.ones(len(d)), Xf])
        X_cap = np.column_stack([np.ones(len(d)), d.logp.astype(float).values])
        rss = lambda X: float(np.sum((yv - X @ np.linalg.lstsq(X, yv, rcond=None)[0]) ** 2))
        tss = float(np.sum((yv - yv.mean()) ** 2))
        r_full, r_fam, r_cap = rss(X_full), rss(X_fam), rss(X_cap)
        df_res = len(d) - X_full.shape[1]
        # type-II partial F tests
        k_f = Xf.shape[1]
        F_fam = ((r_cap - r_full) / k_f) / (r_full / df_res) if df_res > 0 else np.nan
        F_cap = ((r_fam - r_full) / 1) / (r_full / df_res) if df_res > 0 else np.nan
        anova[y] = dict(n=len(d), R2_full=1 - r_full / tss, R2_family_only=1 - r_fam / tss, R2_capacity_only=1 - r_cap / tss,
                        partial_eta2_family=(r_cap - r_full) / r_cap if r_cap > 0 else np.nan,
                        partial_eta2_capacity=(r_fam - r_full) / r_fam if r_fam > 0 else np.nan,
                        p_family=float(stats.f.sf(F_fam, k_f, df_res)) if df_res > 0 else np.nan,
                        p_capacity=float(stats.f.sf(F_cap, 1, df_res)) if df_res > 0 else np.nan)
    S["capacity_anova"] = anova
    lines = []
    for fam in ["CNN", "Isotropic", "Hierarchical", "Multi-axis", "Hybrid", "Linear"]:
        for mo, o in sorted(cap.items(), key=lambda kv: kv[1]["params"]):
            if o["family"] != fam:
                continue
            lines.append(f"{NAME.get(mo, mo)} & {fam} & {o['params']:.1f} & {100 * o['top1']:.1f} & "
                         f"{f2(o.get('pg_grad_cam'))} & {f2(o.get('ebpg_grad_cam'))} & {f2(o.get('ebpg_grad_cam_plus_plus'))} & "
                         f"{f2(o.get('ebpg_smoothgrad'))} & {f2(o.get('ms_grad_cam_med'))} & {f2(o.get('ms_grad_cam_plus_plus_med'))} \\\\")
    save_tex("tab_capacity.tex", "\n".join(lines))


def _json(o):
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    return str(o)


json.dump(S, open(os.path.join(A, "summary.json"), "w"), indent=1, default=_json)
print("wrote", os.path.join(A, "summary.json"), "keys:", list(S.keys()))

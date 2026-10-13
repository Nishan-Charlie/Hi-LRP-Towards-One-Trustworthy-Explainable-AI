"""TEST-ONLY reference: a minimal 2-stage hierarchical Swin with hand-written LRP.

This exists solely so the conservation tests can exercise the HiLRP hierarchical
rules (patch merge + shifted window) inside a realistic composition. The
attention-block internals here use the *naive* rules from `rules.py`, which are
numerically unstable once biases are on (documented in the tests). Production
attribution inherits LXT's stabilized block rules and drops in only the
`hierarchical.py` rules -- do NOT use this model to attribute real networks.
"""
import torch
import torch.nn.functional as F

from .rules import (
    lrp_linear, lrp_layernorm_identity, lrp_softmax_identity,
    bilinear_lrp, residual_lrp, layernorm,
)
from .hierarchical import (
    window_partition, window_reverse, cyclic_shift, cyclic_unshift,
    make_attn_mask, patch_merge_forward, patch_merge_lrp,
)


class BlockParams:
    """Random weights for one Swin block (single head; multi-head is a conserving
    reshape and adds no risk)."""
    def __init__(self, dim, mlp_ratio=2, bias=True, gen=None):
        g = lambda *sz: torch.randn(*sz, generator=gen) * (sz[-1] ** -0.5)
        rn = lambda *sz: torch.randn(*sz, generator=gen)
        self.dim = dim
        self.ln1_g = torch.ones(dim); self.ln1_b = (rn(dim) * .3 if bias else torch.zeros(dim))
        self.ln2_g = torch.ones(dim); self.ln2_b = (rn(dim) * .3 if bias else torch.zeros(dim))
        self.Wqkv = g(3 * dim, dim); self.bqkv = (rn(3 * dim) * .1 if bias else None)
        self.Wproj = g(dim, dim);    self.bproj = (rn(dim) * .1 if bias else None)
        h = dim * mlp_ratio
        self.Wfc1 = g(h, dim); self.bfc1 = (rn(h) * .1 if bias else None)
        self.Wfc2 = g(dim, h); self.bfc2 = (rn(dim) * .1 if bias else None)


# ----------------------------------------------------------- window attention
def attn_forward(x, p, ws, shift, mask):
    H, W, C = x.shape
    scale = C ** -0.5
    if shift:
        x = cyclic_shift(x, shift)
    win = window_partition(x, ws)
    qkv = win @ p.Wqkv.T + (0 if p.bqkv is None else p.bqkv)
    q, k, v = qkv.chunk(3, dim=-1)
    s = (q @ k.transpose(-2, -1)) * scale
    if mask is not None:
        s = s + mask
    prob = torch.softmax(s, dim=-1)
    o = prob @ v
    o = o @ p.Wproj.T + (0 if p.bproj is None else p.bproj)
    o = window_reverse(o, ws, H, W)
    if shift:
        o = cyclic_unshift(o, shift)
    cache = dict(win=win, q=q, k=k, v=v, prob=prob, scale=scale, H=H, W=W)
    return o, cache


def attn_lrp(R_o, p, ws, shift, cache, eps):
    H, W = cache["H"], cache["W"]
    if shift:
        R_o = cyclic_shift(R_o, shift)
    R_ow = window_partition(R_o, ws)
    o_before_proj = cache["prob"] @ cache["v"]
    R_pv = lrp_linear(o_before_proj, p.Wproj, p.bproj, R_ow, eps=eps)
    R_prob, R_v = bilinear_lrp(cache["prob"], cache["v"], R_pv, eps=eps)     # attn @ v
    R_s = lrp_softmax_identity(R_prob)
    R_q, R_kT = bilinear_lrp(cache["q"] * cache["scale"],
                             cache["k"].transpose(-2, -1), R_s, eps=eps)     # q @ k^T
    R_k = R_kT.transpose(-2, -1)
    R_qkv = torch.cat([R_q, R_k, R_v], dim=-1)
    R_win = lrp_linear(cache["win"], p.Wqkv, p.bqkv, R_qkv, eps=eps)
    R_x = window_reverse(R_win, ws, H, W)
    if shift:
        R_x = cyclic_unshift(R_x, shift)
    return R_x, R_prob                                    # R_prob returned for mask-leak check


# --------------------------------------------------------------------- MLP
def mlp_forward(x, p):
    return F.gelu(x @ p.Wfc1.T + (0 if p.bfc1 is None else p.bfc1)) @ p.Wfc2.T \
        + (0 if p.bfc2 is None else p.bfc2)


def mlp_lrp(x, p, R_out, eps):
    h = F.gelu(x @ p.Wfc1.T + (0 if p.bfc1 is None else p.bfc1))
    R_h = lrp_linear(h, p.Wfc2, p.bfc2, R_out, eps=eps)
    R_pre = R_h                                           # GELU identity pass-through
    return lrp_linear(x, p.Wfc1, p.bfc1, R_pre, eps=eps)


# ------------------------------------------------------------------- block
def block_forward(x, p, ws, shift, mask):
    a, cache = attn_forward(layernorm(x, p.ln1_g, p.ln1_b, 1e-5), p, ws, shift, mask)
    x1 = x + a
    m = mlp_forward(layernorm(x1, p.ln2_g, p.ln2_b, 1e-5), p)
    x2 = x1 + m
    return x2, (cache, x, x1, a, m)


def block_lrp(R_out, p, ws, shift, saved, eps):
    cache, x0, x1, a, m = saved
    R_x1, R_m = residual_lrp(x1, m, R_out, eps=eps)
    R_x1 = R_x1 + lrp_layernorm_identity(mlp_lrp(layernorm(x1, p.ln2_g, p.ln2_b, 1e-5), p, R_m, eps))
    R_x0, R_a = residual_lrp(x0, a, R_x1, eps=eps)
    R_aln, R_prob = attn_lrp(R_a, p, ws, shift, cache, eps)
    R_x0 = R_x0 + lrp_layernorm_identity(R_aln)
    return R_x0, R_prob


# ----------------------------------------------------------------- full model
class ToyModel:
    def __init__(self, H=8, C=16, ws=4, nclass=5, bias=True, seed=0):
        gen = torch.Generator().manual_seed(seed)
        rn = lambda *sz: torch.randn(*sz, generator=gen)
        self.H = H; self.C = C; self.ws = ws; self.shift = ws // 2
        self.Wembed = rn(C, 3) * 0.3; self.bembed = (rn(C) * .1 if bias else None)
        self.p1a = BlockParams(C, bias=bias, gen=gen)
        self.p1b = BlockParams(C, bias=bias, gen=gen)
        self.mg_g = torch.ones(4 * C); self.mg_b = (rn(4 * C) * .3 if bias else torch.zeros(4 * C))
        self.mg_W = rn(2 * C, 4 * C) * (4 * C) ** -0.5
        self.p2 = BlockParams(2 * C, bias=bias, gen=gen)
        self.Whead = rn(nclass, 2 * C) * (2 * C) ** -0.5
        self.bhead = (rn(nclass) * .1 if bias else None)
        self.mask = make_attn_mask(H, W=H, ws=ws, shift=self.shift)

    def forward(self, px):
        C = self.C
        x = px @ self.Wembed.T + (0 if self.bembed is None else self.bembed)
        x, s1a = block_forward(x, self.p1a, self.ws, 0, None)
        x, s1b = block_forward(x, self.p1b, self.ws, self.shift, self.mask)
        y, zmg = patch_merge_forward(x, self.mg_g, self.mg_b, 1e-5, self.mg_W)
        y, s2 = block_forward(y, self.p2, self.ws, 0, None)
        pooled = y.reshape(-1, 2 * C).mean(0)
        logits = pooled @ self.Whead.T + (0 if self.bhead is None else self.bhead)
        return logits, dict(y=y, pooled=pooled, zmg=zmg, s1a=s1a, s1b=s1b, s2=s2, px=px)

    def explain(self, logits, cache, eps=1e-9, target=None):
        """Returns (R_input, diagnostics). diagnostics has total, end-to-end relerr,
        and the SW-MSA mask leak."""
        C = self.C
        if target is None:
            target = int(logits.argmax())
        R_logits = torch.zeros_like(logits); R_logits[target] = logits[target]
        R0 = R_logits.sum()

        R_pooled = lrp_linear(cache["pooled"], self.Whead, self.bhead, R_logits, eps=eps)
        flat = cache["y"].reshape(-1, 2 * C)
        denom = flat.sum(0, keepdim=True)
        denom = denom + eps * torch.sign(denom)
        R_y = (flat / denom * R_pooled).reshape(cache["y"].shape)

        R_y, _ = block_lrp(R_y, self.p2, self.ws, 0, cache["s2"], eps)
        R_premerge = R_y.sum().item()                         # relevance entering the merge
        R_x = patch_merge_lrp(R_y, cache["zmg"], self.mg_g, self.mg_b, 1e-5, self.mg_W, C, eps)
        R_postmerge = R_x.sum().item()                        # relevance leaving the merge
        R_x, R_probSW = block_lrp(R_x, self.p1b, self.ws, self.shift, cache["s1b"], eps)
        R_x, _ = block_lrp(R_x, self.p1a, self.ws, 0, cache["s1a"], eps)
        R_px = lrp_linear(cache["px"], self.Wembed, self.bembed, R_x, eps=eps)

        # mask-leak diagnostic only applies when the SW-MSA mask exists
        # (the cyclic/idealized variant used by the equivariance tests sets mask=None)
        leak = R_probSW[self.mask < -1].abs().sum().item() if self.mask is not None else 0.0
        relerr = abs(R_px.sum().item() - R0.item()) / abs(R0.item())
        # local conservation of the patch-merge rule *in context* (its own contribution,
        # independent of the naive block internals feeding it)
        merge_relerr = abs(R_postmerge - R_premerge) / abs(R_premerge)
        return R_px, dict(total=R0.item(), relerr=relerr, mask_leak=leak,
                          merge_relerr=merge_relerr)

"""Attention as four conservation-preserving primitives.

The central generalization claim of HiLRP: every attention mechanism, however
exotic, is a composition of four operation classes, each with one conservation
rule, so HiLRP propagates relevance through any attention.

    1. Linear map        Q/K/V/out projections, poolings, kernel maps  -> lrp_linear
    2. Bilinear mix      the matmuls (QK^T, attn.V, linear-attn prods)  -> bilinear_lrp
    3. Normalization/gate softmax, linear-attn denom, sigmoid/SE gate   -> detached-denominator / detached-gate identity
    4. Reindexing        heads, windows, shifts, dilation, groups       -> permutation (exact, Lemma 2)

This module gives minimal, bias-free, float64 reference implementations
(forward + relevance-backward) of the composite attention TYPES built from these
primitives, so `tests/test_attention_coverage.py` can verify that each conserves
relevance to machine precision. These are reference/toy implementations for the
guarantee; production attribution on real models inherits LXT's stabilized block
rules and the per-family attention flag (see the *_lxt.py ports).

Every function below uses eps=0 and no biases, so conservation is exact up to
float64 round-off.
"""
import torch

from .rules import lrp_linear, lrp_softmax_identity, bilinear_lrp, residual_lrp


# --------------------------------------------------------------- primitive: gate
def gate_lrp(feature, gate, R_out):
    """Detached-gate rule for an elementwise gated product out = feature * gate,
    where `gate` is the attention weight (channel/coordinate/SE attention).

    Treating the gate as fixed mixing (CP principle), relevance flows entirely to
    the feature path: R_feature = R_out (since out_i = feature_i * gate_i, the
    contribution ratio is 1). This is why channel-attention relevance passes
    through the feature, not the gate branch. Conserves exactly.
    """
    return R_out.clone()


def _matmul_detached_lrp(M, v, R_out, eps=0.0):
    """Relevance for out = M @ v with M DETACHED (M is a fixed mixing matrix, e.g.
    a linear-attention kernel or a CP-detached softmax). All relevance flows to v.
    sum(R_v) == sum(R_out)."""
    out = M @ v
    s = R_out / (out + eps * torch.sign(out) + (out == 0) * 1e-30)
    return (M.transpose(-2, -1) @ s) * v


# ------------------------------------------------------- self / multi-head (1 head)
def self_attention(x, Wq, Wk, Wv, Wo):
    q, k, v = x @ Wq.T, x @ Wk.T, x @ Wv.T
    scale = q.shape[-1] ** -0.5
    p = torch.softmax((q @ k.T) * scale, dim=-1)
    o = p @ v
    return o @ Wo.T, dict(x=x, q=q, k=k, v=v, p=p, o=o, scale=scale)


def self_attention_lrp(cache, W, R_out, eps=0.0):
    Wq, Wk, Wv, Wo = W
    R_o = lrp_linear(cache["o"], Wo, None, R_out, eps=eps)
    R_p, R_v = bilinear_lrp(cache["p"], cache["v"], R_o, eps=eps)
    R_s = lrp_softmax_identity(R_p)
    R_q, R_kT = bilinear_lrp(cache["q"] * cache["scale"], cache["k"].T, R_s, eps=eps)
    R_k = R_kT.T
    x = cache["x"]
    return (lrp_linear(x, Wq, None, R_q, eps=eps)
            + lrp_linear(x, Wk, None, R_k, eps=eps)
            + lrp_linear(x, Wv, None, R_v, eps=eps))


# ------------------------------------------------------------------ cross-attention
def cross_attention(xa, xb, Wq, Wk, Wv, Wo):
    """Query from stream a, key/value from stream b (encoder-decoder / bi-modal)."""
    q, k, v = xa @ Wq.T, xb @ Wk.T, xb @ Wv.T
    scale = q.shape[-1] ** -0.5
    p = torch.softmax((q @ k.T) * scale, dim=-1)
    o = p @ v
    return o @ Wo.T, dict(xa=xa, xb=xb, q=q, k=k, v=v, p=p, o=o, scale=scale)


def cross_attention_lrp(cache, W, R_out, eps=0.0):
    """Returns (R_xa, R_xb): relevance splits across the two input streams."""
    Wq, Wk, Wv, Wo = W
    R_o = lrp_linear(cache["o"], Wo, None, R_out, eps=eps)
    R_p, R_v = bilinear_lrp(cache["p"], cache["v"], R_o, eps=eps)
    R_s = lrp_softmax_identity(R_p)
    R_q, R_kT = bilinear_lrp(cache["q"] * cache["scale"], cache["k"].T, R_s, eps=eps)
    R_k = R_kT.T
    R_xa = lrp_linear(cache["xa"], Wq, None, R_q, eps=eps)
    R_xb = (lrp_linear(cache["xb"], Wk, None, R_k, eps=eps)
            + lrp_linear(cache["xb"], Wv, None, R_v, eps=eps))
    return R_xa, R_xb


# --------------------------------------------------------- co-attention (bi-modal)
def co_attention(xa, xb, Wa, Wb):
    """Two parallel cross-attentions with residual updates, as in VQA co-attention:
    each modality attends to the other and updates itself."""
    ca, cache_a = cross_attention(xa, xb, *Wa)
    cb, cache_b = cross_attention(xb, xa, *Wb)
    out_a, out_b = xa + ca, xb + cb
    return (out_a, out_b), dict(xa=xa, xb=xb, ca=ca, cb=cb, ca_cache=cache_a, cb_cache=cache_b)


def co_attention_lrp(cache, Wa, Wb, R_out_a, R_out_b, eps=0.0):
    R_xa1, R_ca = residual_lrp(cache["xa"], cache["ca"], R_out_a, eps=eps)
    R_xb1, R_cb = residual_lrp(cache["xb"], cache["cb"], R_out_b, eps=eps)
    R_xa2, R_xb2 = cross_attention_lrp(cache["ca_cache"], Wa, R_ca, eps=eps)   # a<-b
    R_xb3, R_xa3 = cross_attention_lrp(cache["cb_cache"], Wb, R_cb, eps=eps)   # b<-a
    return R_xa1 + R_xa2 + R_xa3, R_xb1 + R_xb2 + R_xb3


# --------------------------------------------------------------- linear attention
def linear_attention(x, Wq, Wk, Wv, Wo, kernel=None):
    """Softmax-free linear attention: out = phi(q) (phi(k)^T v) / (phi(q) . sum phi(k)).
    Used by EfficientViT/Performer. The mixing is a detached normalization."""
    kernel = kernel or (lambda t: torch.relu(t) + 1.0)
    q, k, v = x @ Wq.T, x @ Wk.T, x @ Wv.T
    aq, ak = kernel(q), kernel(k)
    num = aq @ (ak.T @ v)                      # [N, d]
    den = aq @ ak.sum(0, keepdim=True).T       # [N, 1]
    o = num / den
    return o @ Wo.T, dict(x=x, v=v, o=o, M=(aq @ ak.T) / den)   # effective mixing M (detached)


def linear_attention_lrp(cache, W, R_out, eps=0.0):
    Wq, Wk, Wv, Wo = W
    R_o = lrp_linear(cache["o"], Wo, None, R_out, eps=eps)
    R_v = _matmul_detached_lrp(cache["M"].detach(), cache["v"], R_o, eps=eps)  # CP: mix detached
    return lrp_linear(cache["x"], Wv, None, R_v, eps=eps)


# ------------------------------------------------ channel / coordinate (gated) attention
def channel_attention(x, W1, W2):
    """Squeeze-excite style channel attention: gate = sigmoid(W2 relu(W1 mean(x)));
    out = x * gate. Coordinate attention is the same with directional pools."""
    g = torch.sigmoid(torch.relu(x.mean(0) @ W1.T) @ W2.T)   # [C]
    return x * g, dict(x=x, g=g)


def channel_attention_lrp(cache, R_out, eps=0.0):
    """CP: the gate is detached mixing, so relevance flows through the feature."""
    return gate_lrp(cache["x"], cache["g"], R_out)


# ------------------------------------------------------- dilated / sparse attention
def dilated_attention(x, Wq, Wk, Wv, Wo, dilation=2):
    """Attention over a dilated (strided) subset of tokens; the rest pass through.
    Gather/scatter are selections (Lemma 2)."""
    idx = torch.arange(0, x.shape[0], dilation)
    sub = x[idx]
    attn_out, cache = self_attention(sub, Wq, Wk, Wv, Wo)
    out = x.clone()
    out[idx] = attn_out
    return out, dict(idx=idx, sub_cache=cache, N=x.shape[0])


def dilated_attention_lrp(cache, W, R_out, eps=0.0):
    idx = cache["idx"]
    R_x = R_out.clone()                        # non-attended tokens: identity pass-through
    R_sub = self_attention_lrp(cache["sub_cache"], W, R_out[idx], eps=eps)
    R_x[idx] = R_sub                           # attended subset: through attention
    return R_x


# ------------------------------------------------ deformable attention (sampling)
def _bilinear_1d(L, loc):
    """Interpolation weights over L grid points for a continuous loc in [0, L-1].
    Two nonzero entries (floor, ceil) summing to 1: bilinear sampling is a LINEAR
    map on the value grid, so it conserves (Lemma 1, one row per sample)."""
    loc = loc.clamp(0, L - 1)
    lo = torch.floor(loc).long()
    hi = torch.clamp(lo + 1, max=L - 1)
    w = loc - lo.to(loc.dtype)
    row = torch.zeros(L, dtype=loc.dtype)
    row[lo] += (1 - w)
    row[hi] += w
    return row


def deformable_attention(x, Wv, Wo, Woff, Wattn, scale=0.1):
    """Deformable attention (toy, 1D, as in Deformable-DETR/DAT). Each query
    predicts K sampling offsets and K attention weights; values are bilinearly
    sampled at ref+offset and combined. The offsets/weights build a DETACHED,
    row-stochastic mixing M over value tokens (sampling is linear, the K weights
    sum to 1), so out = M @ (x Wv^T) is exactly the detached-mixing pattern of
    linear attention and conserves. Woff, Wattn: [K, C]; Wv, Wo: [C, C]."""
    v = x @ Wv.T
    L = v.shape[0]
    N = x.shape[0]
    offsets = (x @ Woff.T) * scale                 # [N, K] learned sampling offsets
    attn = torch.softmax(x @ Wattn.T, dim=-1)      # [N, K] weights, sum_k = 1
    refs = torch.linspace(0, L - 1, N)
    K = offsets.shape[1]
    M = torch.zeros(N, L, dtype=x.dtype)
    for i in range(N):
        for k in range(K):
            M[i] += attn[i, k] * _bilinear_1d(L, refs[i] + offsets[i, k])
    o = M @ v                                      # row-stochastic mixing (sum_j M_ij = 1)
    return o @ Wo.T, dict(x=x, v=v, o=o, M=M)


def deformable_attention_lrp(cache, W, R_out, eps=0.0):
    """CP rule: the sampling grid M (offsets x weights) is detached mixing, so
    relevance flows through the value path and back to x via Wv/Wo. Conserves."""
    Wv, Wo = W
    R_o = lrp_linear(cache["o"], Wo, None, R_out, eps=eps)
    R_v = _matmul_detached_lrp(cache["M"].detach(), cache["v"], R_o, eps=eps)
    return lrp_linear(cache["x"], Wv, None, R_v, eps=eps)

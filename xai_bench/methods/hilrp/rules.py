"""Generic LRP relevance-propagation primitives used by HiLRP.

These are the *inherited* rules (AttnLRP / Ali et al. 2022 lineage), not the
contribution. The HiLRP contribution lives in `hierarchical.py`. They are kept
here, self-contained and dtype-honest, only so the conservation tests can
exercise the hierarchical rules end-to-end without pulling in LXT.

    In production, the attention-block internals (qkv/proj/mlp linears, softmax,
    the two attention matmuls) MUST be handled by LXT's *stabilized* rules, which
    correctly manage bias terms and near-zero bilinear denominators. The naive
    versions here are numerically treacherous with biases on (see the `toy_swin`
    reference model and the documented negative test) and exist for testing the
    hierarchical rules in isolation, not for attributing real models.

Design note (unifying): LayerNorm and softmax are both handled by "detach the
normalizing denominator", which makes them locally linear and thus
conservation-preserving. See `lrp_layernorm_identity` for the LN caveat.
"""
import torch

DEFAULT_EPS = 1e-9


def _stabilize(z, eps):
    """Signed-epsilon stabilizer: keeps the denominator away from zero."""
    return z + eps * torch.sign(z)


def layernorm(z, weight, bias, ln_eps):
    """Standard LayerNorm forward over the last axis (for building/inverting)."""
    mu = z.mean(-1, keepdim=True)
    var = z.var(-1, unbiased=False, keepdim=True)
    return (z - mu) / torch.sqrt(var + ln_eps) * weight + bias


def lrp_linear(x, W, b, R_y, eps=DEFAULT_EPS):
    """epsilon-rule for y = x W^T + b.

    Conserves exactly when b is None/0 and eps -> 0; otherwise the bias absorbs
    its own share of relevance (a *bounded* leak), and eps introduces an O(eps)
    leak. Both are the standard, intended LRP behavior.
    """
    z = x @ W.T + (0 if b is None else b)
    s = R_y / _stabilize(z, eps)
    return (s @ W) * x


def lrp_layernorm_identity(R_a):
    """LayerNorm rule (Ali et al. 2022 / AttnLRP): pass relevance through unchanged.

    CAVEAT (conservation is not faithfulness): this rule conserves *exactly* by
    construction (R_z == R_a, same shape). But it is a *stronger* simplification
    than merely detaching the standardization denominator: it also discards the
    effect of the affine scale gamma on how relevance is routed across channels.
    We choose it because the alternative (`lrp_layernorm_affine`) makes the LN
    bias beta a relevance sink and destroys conservation on trained-scale params
    (relerr ~1.1 -- see the negative test). The *faithfulness* cost of the
    identity simplification, if any, is not visible in a conservation test and
    must be checked separately at Gate-2-real (maps land on objects,
    class-sensitive).
    """
    return R_a.clone()


def lrp_layernorm_affine(z, weight, bias, ln_eps, R_a, eps=DEFAULT_EPS):
    """REJECTED alternative LN rule: detach sigma but keep the affine (incl. beta)
    in the epsilon denominator. Kept ONLY to document why we do not use it -- with
    beta != 0 the bias becomes a relevance sink and conservation fails
    (relerr ~1.1). Do not use in anger; see `lrp_layernorm_identity`.
    """
    n = z.shape[-1]
    sigma = torch.sqrt(z.var(-1, unbiased=False, keepdim=True) + ln_eps)
    a = layernorm(z, weight, bias, ln_eps)          # forward incl. beta
    s = R_a / _stabilize(a, eps)
    gs = weight * s
    term = (gs - gs.sum(-1, keepdim=True) / n) / sigma
    return z * term


def lrp_softmax_identity(R_p):
    """Softmax rule: detach the partition function Z -> elementwise -> pass through.

    Same principle as the LayerNorm identity rule (detach the normalizer).
    Conserves exactly. Quality-optimized softmax rules (AttnLRP's Taylor rule)
    are a faithfulness upgrade handled by LXT in production.
    """
    return R_p.clone()


def bilinear_lrp(A, B, R_C, eps=DEFAULT_EPS):
    """Uniform bilinear rule for C = A @ B: split relevance 0.5 to each operand.

    Sum of the two shares is R_C at every output element, so it conserves. Used
    for the two attention matmuls (Q@K^T and attn@V) in the reference model only.
    """
    C = A @ B
    S = 0.5 * R_C / _stabilize(C, eps)
    R_A = (S @ B.transpose(-2, -1)) * A
    R_B = (A.transpose(-2, -1) @ S) * B
    return R_A, R_B


def residual_lrp(a, b, R, eps=DEFAULT_EPS):
    """Split relevance across a residual sum z = a + b, proportional to contribution."""
    d = _stabilize(a + b, eps)
    return a / d * R, b / d * R

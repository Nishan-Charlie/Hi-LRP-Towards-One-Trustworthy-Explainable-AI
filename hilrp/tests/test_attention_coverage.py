"""Conservation across the attention taxonomy (the generalization guarantee).

HiLRP's central generalization claim is that any attention mechanism is a
composition of four conserving primitives (linear, bilinear, normalization/gate,
reindexing), so relevance conserves through any attention. This suite verifies
that claim at machine precision (float64, bias-free, eps=0) for a representative
span of attention TYPES: self / multi-head, cross (encoder-decoder), co-attention
(bi-modal), linear (softmax-free), channel/coordinate (gated), and dilated/sparse.

Each test seeds relevance at the attention output and checks that the total
relevance at the input(s) equals it. For multi-stream attention (cross, co) the
check also confirms relevance is split across streams, not lost.

These are the same guarantees, and the same ceilings, as tests/test_conservation.
"""
import torch
import pytest

from xai_bench.methods.hilrp.attention_primitives import (
    self_attention, self_attention_lrp,
    cross_attention, cross_attention_lrp,
    co_attention, co_attention_lrp,
    linear_attention, linear_attention_lrp,
    channel_attention, channel_attention_lrp,
    dilated_attention, dilated_attention_lrp,
    deformable_attention, deformable_attention_lrp,
)

TOL = 1e-11   # float64, bias-free, eps=0


@pytest.fixture(autouse=True)
def _f64():
    prev = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_default_dtype(prev)


def relerr(a, b):
    return abs(float(a) - float(b)) / (abs(float(b)) + 1e-30)


def _weights(C, n=4, seed=0):
    g = torch.Generator().manual_seed(seed)
    return [torch.randn(C, C, generator=g) * (C ** -0.5) for _ in range(n)]


def test_self_attention_conserves():
    torch.manual_seed(0)
    N, C = 16, 24
    x = torch.randn(N, C)
    W = _weights(C)
    out, cache = self_attention(x, *W)
    R_out = out.clone()
    R_x = self_attention_lrp(cache, W, R_out)
    assert relerr(R_x.sum(), R_out.sum()) < TOL


def test_cross_attention_conserves_and_splits_streams():
    torch.manual_seed(1)
    C = 24
    xa, xb = torch.randn(12, C), torch.randn(20, C)
    W = _weights(C)
    out, cache = cross_attention(xa, xb, *W)
    R_out = out.clone()
    R_xa, R_xb = cross_attention_lrp(cache, W, R_out)
    assert relerr(R_xa.sum() + R_xb.sum(), R_out.sum()) < TOL   # total conserves
    assert R_xa.abs().sum() > 0 and R_xb.abs().sum() > 0        # both streams get relevance


def test_co_attention_bimodal_conserves():
    torch.manual_seed(2)
    C = 20
    xa, xb = torch.randn(10, C), torch.randn(14, C)
    Wa, Wb = _weights(C, seed=3), _weights(C, seed=4)
    (out_a, out_b), cache = co_attention(xa, xb, Wa, Wb)
    R_a, R_b = out_a.clone(), out_b.clone()
    R_xa, R_xb = co_attention_lrp(cache, Wa, Wb, R_a, R_b)
    assert relerr(R_xa.sum() + R_xb.sum(), R_a.sum() + R_b.sum()) < TOL


def test_linear_attention_conserves():
    torch.manual_seed(5)
    N, C = 16, 24
    x = torch.randn(N, C)
    W = _weights(C)
    out, cache = linear_attention(x, *W)
    R_out = out.clone()
    R_x = linear_attention_lrp(cache, W, R_out)
    assert relerr(R_x.sum(), R_out.sum()) < TOL


def test_channel_gated_attention_conserves():
    torch.manual_seed(6)
    N, C = 16, 24
    x = torch.randn(N, C)
    W1, W2 = torch.randn(C, C) * C ** -0.5, torch.randn(C, C) * C ** -0.5
    out, cache = channel_attention(x, W1, W2)
    R_out = out.clone()
    R_x = channel_attention_lrp(cache, R_out)
    assert relerr(R_x.sum(), R_out.sum()) < TOL


def test_dilated_sparse_attention_conserves():
    torch.manual_seed(7)
    N, C = 24, 24
    x = torch.randn(N, C)
    W = _weights(C)
    out, cache = dilated_attention(x, *W, dilation=3)
    R_out = out.clone()
    R_x = dilated_attention_lrp(cache, W, R_out)
    assert relerr(R_x.sum(), R_out.sum()) < TOL


def test_deformable_attention_conserves():
    torch.manual_seed(8)
    N, C, K = 16, 24, 3
    x = torch.randn(N, C)
    Wv, Wo = _weights(C, n=2, seed=8)
    Woff = torch.randn(K, C) * C ** -0.5      # offset generator
    Wattn = torch.randn(K, C) * C ** -0.5     # sampling-weight generator
    out, cache = deformable_attention(x, Wv, Wo, Woff, Wattn)
    # M is row-stochastic (bilinear sample weights x attention weights sum to 1)
    assert relerr(cache["M"].sum(-1).mean(), 1.0) < TOL
    R_out = out.clone()
    R_x = deformable_attention_lrp(cache, (Wv, Wo), R_out)
    assert relerr(R_x.sum(), R_out.sum()) < TOL

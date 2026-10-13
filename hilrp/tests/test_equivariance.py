"""Equivariance guarantees for HiLRP (the conditional theorem, validated where
its premise holds exactly).

THE THEOREM (conditional): if the forward pass commutes with a permutation of
the token grid, then HiLRP relevance commutes with the same permutation.
Proof mechanism: every propagation rule is a function of local activations and
shared weights only (no absolute-position quantities), and permutations conserve
relevance exactly with ratio 1 (Lemma 2), so the backward composition inherits
the forward's symmetry.

REAL Swin does NOT satisfy the premise for any nontrivial translation: the
SW-MSA masks anchor to the canvas, and even mask-free Swin-T has no nontrivial
exact translation symmetry because the odd window grid (7) and the even merge
grid (2) only align at the full image period. The premise is therefore
validated on an idealized cyclic toy where the grids align by design:
window 4, merge 2, H = 16 tokens, no attention masks (cyclic SW-MSA). There,
8-token shifts are a genuine nontrivial symmetry:

  * forward:   logits(shift(x)) == logits(x) to float64 precision
  * relevance: R(shift(x)) == shift(R(x)) to float64 precision

A negative control asserts the symmetry group is the right one: at a 3-token
shift (aligned with neither window nor merge grid), the forward itself changes
and equivariance fails. The attribution tracks the model, not the image.

These tests run in the relevance-direct float64 framework (no LXT, no zennit),
the same instrument that carries the conservation guarantees.
"""
import torch
import pytest

from xai_bench.methods.hilrp.toy_swin import ToyModel

TOL_FORWARD = 1e-10     # float64 forward equivariance
TOL_RELEVANCE = 1e-8    # float64 relevance equivariance (eps-rule leakage)


@pytest.fixture(autouse=True)
def _f64():
    prev = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_default_dtype(prev)


def build_cyclic_toy():
    """Idealized cyclic toy: H=16, window 4, merge 2, NO SW-MSA mask.
    lcm(window, 2 * window) = 8 divides 16, so an 8-token shift is a
    nontrivial exact symmetry of the forward."""
    model = ToyModel(H=16, C=16, ws=4, bias=True, seed=0)
    model.mask = None           # cyclic SW-MSA: shift without canvas masking
    return model


def shift_hw(t, s):
    return torch.roll(t, shifts=(s, s), dims=(0, 1))


def run(model, px, eps=1e-9):
    logits, cache = model.forward(px)
    target = int(logits.argmax())
    R, diag = model.explain(logits, cache, eps=eps, target=target)
    return logits, R, target


def test_forward_equivariant_at_aligned_shift():
    model = build_cyclic_toy()
    px = torch.randn(16, 16, 3)
    l0, _, _ = run(model, px)
    l1, _, _ = run(model, shift_hw(px, 8))
    assert (l1 - l0).abs().max().item() < TOL_FORWARD


def test_relevance_equivariant_at_aligned_shift():
    """The conditional theorem's conclusion, at machine precision: the premise
    (forward commutes) holds exactly at 8-token shifts, so relevance must too."""
    model = build_cyclic_toy()
    px = torch.randn(16, 16, 3)
    l0, R0, t0 = run(model, px)
    l1, R1, t1 = run(model, shift_hw(px, 8))
    assert t0 == t1
    err = (R1 - shift_hw(R0, 8)).abs().max().item() / (R0.abs().max().item() + 1e-30)
    assert err < TOL_RELEVANCE


def test_negative_control_unaligned_shift_breaks_forward():
    """Symmetry-group sanity: a 3-token shift aligns with neither the window
    nor the merge grid, so the FORWARD itself changes. If this ever passes as
    equivariant, the toy stopped exercising the hierarchy."""
    model = build_cyclic_toy()
    px = torch.randn(16, 16, 3)
    l0, _, _ = run(model, px)
    l1, _, _ = run(model, shift_hw(px, 3))
    assert (l1 - l0).abs().max().item() > 1e-3


def test_masks_break_the_premise():
    """Documentation test for the paper's honesty clause: with the real SW-MSA
    canvas mask, even the aligned shift is no longer a forward symmetry, which
    is exactly why the theorem must be conditional."""
    model = ToyModel(H=16, C=16, ws=4, bias=True, seed=0)   # mask kept
    px = torch.randn(16, 16, 3)
    l0, _, _ = run(model, px)
    l1, _, _ = run(model, shift_hw(px, 8))
    assert (l1 - l0).abs().max().item() > 1e-6

"""Conservation guarantees for the HiLRP hierarchical rules.

WHAT THESE TESTS DO AND DO NOT ASSERT
-------------------------------------
They assert CONSERVATION: that relevance in == relevance out for the
resolution-reduction rules (patch merge, un-concat, shifted-window regroup) and
their bias-free composition. They do NOT assert FAITHFULNESS -- whether the
resulting attribution map is *correct* (lands on the object, is class-sensitive).
Those are different properties:

  * Conservation  = nothing leaked. A permutation of relevance conserves
    perfectly yet can be totally wrong. Guaranteed here, at machine precision.
  * Faithfulness  = the map reflects the model's actual reasoning. NOT testable
    by a sum check. Validated separately at Gate-2-real (real Swin-T via LXT:
    maps land on objects, change with the target class).

In particular the LayerNorm *identity* rule is chosen because it conserves (the
affine-detach alternative makes beta a relevance sink -- see the negative test),
but the identity rule additionally discards gamma's effect on relevance routing.
That simplification's faithfulness cost, if any, is invisible here by design and
must be checked at Gate-2-real. Do not read a passing conservation suite as
evidence the maps are right.

THRESHOLDS ARE DTYPE-TIED. All tests run in float64. The ceilings below are the
actual errors achieved by the verified rules; any refactor that silently breaks
conservation must fail loudly. Re-derive the ceilings if the dtype changes.
"""
import torch
import pytest

from xai_bench.methods.hilrp.hierarchical import (
    patch_merge_forward, patch_merge_lrp, unconcat_2x2,
    window_partition, window_reverse, cyclic_shift, cyclic_unshift,
)
from xai_bench.methods.hilrp.rules import (
    lrp_linear, lrp_layernorm_identity, lrp_layernorm_affine, layernorm,
)
from xai_bench.methods.hilrp.toy_swin import ToyModel

# dtype-tied numeric ceilings (float64) -- the guarantee, as sharp as the proof
TOL_MERGE_PURE     = 1e-13   # patch-merge rule (reduction + identity-LN + un-concat), eps=0
TOL_UNCONCAT       = 1e-14   # un-concat regroup total preserved
TOL_LN_IDENTITY    = 1e-13   # LN identity rule conserves even with beta != 0
TOL_MASK_LEAK      = 1e-30   # relevance on blocked SW-MSA pairs (structural ~1e-43)
TOL_TIMM_FWD       = 1e-14   # our forward vs timm (machine epsilon; op order differs)
TOL_COMPOSE_LOOSE  = 1e-2    # bias-free stack sanity; NOT sharp (naive-block denom leak)


@pytest.fixture(autouse=True)
def _f64():
    prev = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_default_dtype(prev)


def relerr(R_in, R_out):
    return abs(R_in.sum().item() - R_out.sum().item()) / abs(R_out.sum().item())


# ----------------------------------------------------------- patch merge core
def test_patch_merge_pure_rule_conserves():
    """gamma=1, beta=0, eps=0 -> concat->LN->Linear conserves to machine precision."""
    torch.manual_seed(0)
    C, N = 96, 49
    z = torch.randn(N, 4 * C)
    g, b = torch.ones(4 * C), torch.zeros(4 * C)
    Wr = torch.randn(2 * C, 4 * C) * (4 * C) ** -0.5
    a = layernorm(z, g, b, 1e-5)
    R_y = a @ Wr.T                                    # standard LRP init at the output
    R_a = lrp_linear(a, Wr, None, R_y, eps=0.0)
    R_z = lrp_layernorm_identity(R_a)
    assert relerr(R_z, R_y) < TOL_MERGE_PURE


def test_layernorm_identity_conserves_with_beta():
    """The chosen LN rule conserves even at trained-scale beta != 0."""
    torch.manual_seed(1)
    C, N = 96, 49
    z = torch.randn(N, 4 * C)
    g = torch.randn(4 * C) * .3 + 1.0
    b = torch.randn(4 * C) * .3                       # beta != 0
    a = layernorm(z, g, b, 1e-5)
    R_a = a.clone()
    R_z = lrp_layernorm_identity(R_a)
    assert relerr(R_z, R_a) < TOL_LN_IDENTITY


def test_affine_layernorm_leaks_documents_rejection():
    """NEGATIVE test: the rejected affine-detach LN rule loses conservation once
    beta != 0 (beta becomes a relevance sink). This is *why* we use identity."""
    torch.manual_seed(1)
    C, N = 96, 49
    z = torch.randn(N, 4 * C)
    g = torch.randn(4 * C) * .3 + 1.0
    b = torch.randn(4 * C) * .3
    a = layernorm(z, g, b, 1e-5)
    R_a = a.clone()
    R_z = lrp_layernorm_affine(z, g, b, 1e-5, R_a, eps=0.0)
    assert relerr(R_z, R_a) > 0.5                     # catastrophically non-conserving


# --------------------------------------------------------- un-concat / regroup
def test_unconcat_regroup_is_lossfree_and_exact_inverse():
    """The 2x2 regroup round-trips to identity (exact inverse) and preserves the
    relevance total -- a permutation preserves the sum trivially, but this also
    checks it is the *correct* inverse, not just some permutation."""
    torch.manual_seed(2)
    H = W = 14; C = 16
    x = torch.randn(H, W, C)
    grouped = (x.reshape(H // 2, 2, W // 2, 2, C).permute(0, 2, 3, 1, 4)
                .reshape(H // 2, W // 2, 4 * C))
    back = unconcat_2x2(grouped, C)
    assert (back - x).abs().max().item() == 0.0       # exact inverse (round-trip)
    R = torch.randn(H // 2, W // 2, 4 * C)
    assert relerr(unconcat_2x2(R, C), R) < TOL_UNCONCAT


# ------------------------------------------------------ forward matches timm
def test_forward_matches_timm_patchmerging():
    """Our decomposed forward equals timm's PatchMerging to float64 machine
    precision (~1e-15). Not exactly 0.0 only because our clean non-batched op
    order differs from timm's; the result is identical up to round-off."""
    pytest.importorskip("timm")
    from timm.models.swin_transformer import PatchMerging
    H = W = 14; C = 32
    pm = PatchMerging(dim=C).to(torch.float64)
    x = torch.randn(1, H, W, C)
    ours, _ = patch_merge_forward(x[0], pm.norm.weight, pm.norm.bias,
                                  pm.norm.eps, pm.reduction.weight)
    assert (ours - pm(x)[0]).abs().max().item() < TOL_TIMM_FWD


# -------------------------------------------- shifted-window permutation exactness
def test_window_partition_reverse_roundtrip():
    torch.manual_seed(3)
    H = W = 8; C = 16; ws = 4
    x = torch.randn(H, W, C)
    assert (window_reverse(window_partition(x, ws), ws, H, W) - x).abs().max().item() == 0.0

def test_cyclic_shift_roundtrip():
    torch.manual_seed(4)
    H = W = 8; C = 16; shift = 2
    x = torch.randn(H, W, C)
    assert (cyclic_unshift(cyclic_shift(x, shift), shift) - x).abs().max().item() == 0.0


# -------------------------------------------- patch-merge rule in isolation
def test_patch_merge_rule_isolated_conserves():
    """THE contribution, isolated: feed the full patch-merge backward rule
    (reduction + identity-LN + un-concat) an arbitrary known-good relevance on the
    merged tokens and confirm it conserves, with beta != 0 and eps = 0. Does not
    touch the naive block internals -- this is the sharp guarantee."""
    torch.manual_seed(5)
    H = W = 8; C = 16
    x = torch.randn(H, W, C)
    g = torch.randn(4 * C) * .3 + 1.0
    b = torch.randn(4 * C) * .3                       # beta != 0 (identity LN handles it)
    Wr = torch.randn(2 * C, 4 * C) * (4 * C) ** -0.5
    _, z = patch_merge_forward(x, g, b, 1e-5, Wr)
    # positive known-good relevance: keeps the relerr denominator (sum) well away
    # from zero, so the ratio reflects the true leak, not a small-sum artifact
    R_y = torch.rand(H // 2, W // 2, 2 * C)
    R_x = patch_merge_lrp(R_y, z, g, b, 1e-5, Wr, C, eps=0.0)
    assert relerr(R_x, R_y) < TOL_MERGE_PURE


# ------------------------------------------------ end-to-end composition (bias-free)
def test_patch_merge_conserves_in_context():
    """SHARP: inside the full bias-free composition, the patch-merge step conserves
    the relevance it is actually fed by the stack (its own contribution, isolated
    from the naive blocks around it)."""
    model = ToyModel(bias=False, seed=0)
    px = torch.randn(model.H, model.H, 3)
    logits, cache = model.forward(px)
    _, diag = model.explain(logits, cache, eps=0.0)
    assert diag["merge_relerr"] < TOL_MERGE_PURE


def test_end_to_end_biasfree_no_structural_break():
    """LOOSE sanity: the whole bias-free stack composes without a gross/structural
    conservation break. It is NOT machine-precision because the naive test-only
    block rules leak via near-zero bilinear denominators (draw-dependent) -- the
    very instability that makes us inherit LXT's stabilized internals in
    production. The sharp guarantees are the isolated tests above."""
    model = ToyModel(bias=False, seed=0)
    px = torch.randn(model.H, model.H, 3)
    logits, cache = model.forward(px)
    _, diag = model.explain(logits, cache, eps=1e-9)
    assert diag["relerr"] < TOL_COMPOSE_LOOSE


def test_shifted_window_mask_has_no_leak():
    """SW-MSA blocked/masked attention pairs carry ~0 relevance (softmax floor),
    independent of epsilon -> the shift/mask transition is conservation-neutral."""
    torch.manual_seed(0)
    model = ToyModel(bias=False, seed=0)
    px = torch.randn(model.H, model.H, 3)
    logits, cache = model.forward(px)
    _, diag = model.explain(logits, cache, eps=1e-9)
    assert diag["mask_leak"] < TOL_MASK_LEAK


def test_naive_block_rules_are_unstable_with_bias_documented():
    """NEGATIVE / documentation test: the naive attention-block rules are
    numerically unstable with biases on (near-zero bilinear denominators get
    amplified). This is why production inherits LXT's stabilized rules and drops
    in only the hierarchical rules. Asserting the instability keeps that boundary
    from being silently 'fixed' by reimplementing the internals here."""
    torch.manual_seed(0)
    model = ToyModel(bias=True, seed=0)
    px = torch.randn(model.H, model.H, 3)
    logits, cache = model.forward(px)
    _, diag = model.explain(logits, cache, eps=1e-9)
    assert diag["relerr"] > 1.0                        # not our rules' fault; inherit LXT


# ------------------------------------------------ deformable attention sampling
def test_deformable_sampling_rule_conserves():
    """Deformable attention bilinear sampling rule conserves exactly via autograd.
    It treats the sampling grid as fixed routing weights for a linear map."""
    from xai_bench.methods.hilrp.hierarchical import deformable_sample_forward, deformable_sample_lrp
    torch.manual_seed(6)
    B, C, H, W = 2, 3, 4, 4
    x = torch.rand(B, C, H, W) + 1.0  # strictly positive to keep relerr denominator stable
    grid = torch.rand(B, 3, 3, 2) * 2 - 1.0  # normalized coordinates in [-1, 1]
    
    y = deformable_sample_forward(x, grid)
    R_y = torch.rand_like(y)
    
    R_x = deformable_sample_lrp(R_y, x, grid, eps=0.0)
    assert relerr(R_x, R_y) < 1e-13

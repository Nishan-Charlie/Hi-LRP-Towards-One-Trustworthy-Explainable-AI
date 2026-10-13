"""Production-backend checks for the HiLRP ports (LXT efficient backend + zennit).

WHAT THESE TESTS DO AND DO NOT ASSERT
-------------------------------------
``tests/`` covers the float64 REFERENCE rules and a toy model, with no backend.
This suite covers what actually produces the published maps: the ``*_lxt.py``
ports, on real timm architectures, through LXT and zennit. For every port it
asserts

  * forward equivalence   the class-level patches and the composite leave every
                          logit unchanged ("forward numerics identical");
  * a usable map          [H, W], finite, not identically zero;
  * a complete trace      one finite sum(R)/logit per capture;
  * coexistence           a port's map is unchanged after the other ports patch,
                          which is what lets one `hilrp` run sweep several
                          backbones (see xai_bench/isolation.py).

and, ONLY where the rules guarantee it, conservation:

  * Swin and PVT-v2 with bias-free weights conserve at every capture, pixels
    included, to float32 precision. With no bias to absorb relevance any
    deviation from 1 is a wiring error; an unpatched norm subclass, for one,
    drives the sum to 0.

They do NOT assert conservation on the convolution-hybrid ports (EfficientViT,
MobileViT-v2, MaxViT) or on flat ViTs. There the relevance sum drifts even with
bias-free weights, so it is a MEASURED quantity, reported per backbone in the
README, and not a guarantee. Nor do they assert faithfulness; a passing run says
nothing about whether a map is right (see tests/test_conservation.py).

Weights are random (no download, CPU, float32) with non-trivial biases and
BatchNorm statistics, so the BN-merge canonizer and every bias path is exercised
by the forward check. timm's default init leaves most of them at zero.

RUN IN ITS OWN PROCESS. Importing a port patches classes for the life of the
interpreter, so this directory is deliberately outside ``testpaths``:

    python -m pytest xai_bench/methods/hilrp/tests_backend -v

The coexistence check is in-process, so it sees a port before and after LATER
ports patch, never the effect of earlier ones. Each port was also compared
against a fresh interpreter on pretrained weights; that is not automated here.
"""
import importlib

import pytest
import torch

import xai_bench  # noqa: F401  installs the transformers shim; must precede LXT
from xai_bench import isolation

timm = pytest.importorskip("timm")
pytest.importorskip("zennit", reason="HiLRP backend missing: pip install 'lxt==2.1' zennit")
pytest.importorskip("lxt.efficient", reason="HiLRP backend missing: pip install 'lxt==2.1' zennit")

# The smallest timm variant of each family a port covers: (port module, entry point).
PORTS = {
    "swin_tiny_patch4_window7_224": ("swin_lxt", "attribute_swin"),
    "pvt_v2_b0":                    ("pvt_lxt", "attribute_pvt"),
    "maxvit_tiny_tf_224":           ("maxvit_lxt", "attribute_maxvit"),
    "mobilevitv2_050":              ("mobilevit_lxt", "attribute_mobilevit"),
    "efficientvit_b0":              ("efficientvit_lxt", "attribute_efficientvit"),
    "vit_tiny_patch16_224":         ("vit_lxt", "attribute_vit"),
}
CONSERVING = ("swin_tiny_patch4_window7_224", "pvt_v2_b0")   # guaranteed when bias-free

# dtype-tied ceilings (float32); the errors actually achieved are in the comments
TOL_FORWARD  = 1e-5    # max |logit diff| / max |logit|; achieved <= 7e-7 on every port
TOL_CONSERVE = 1e-4    # |sum(R)/logit - 1|, bias-free Swin / PVT; achieved <= 7e-6
TOL_COEXIST  = 1e-6    # max |map diff| / max |map| once all ports patched; achieved 0


def _build(name, bias_free):
    torch.manual_seed(0)
    model = timm.create_model(name, pretrained=False).eval()
    gen = torch.Generator().manual_seed(1)
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                if bias_free:
                    module.running_mean.zero_()
                else:
                    module.running_mean.normal_(0, 0.1, generator=gen)
                    module.running_var.uniform_(0.5, 1.5, generator=gen)
        for pname, param in model.named_parameters():
            if pname.endswith("bias"):
                param.zero_() if bias_free else param.normal_(0, 0.1, generator=gen)
    return model


@pytest.fixture(scope="module")
def runs():
    """Attribute every port once, in an order that keeps the references honest."""
    live = isolation.live_patches()
    if live:
        pytest.fail(f"needs a fresh interpreter: class-level patches already live from {live}")

    x = torch.randn(1, 3, 224, 224, generator=torch.Generator().manual_seed(2))
    models = {name: _build(name, bias_free=False) for name in PORTS}
    bias_free = {name: _build(name, bias_free=True) for name in CONSERVING}

    with torch.no_grad():                       # ALL references before ANY patch
        refs = {name: model(x) for name, model in models.items()}
    assert isolation.live_patches() == []

    out = {}
    for name, (module, entry) in PORTS.items():
        attribute = getattr(importlib.import_module(f"xai_bench.methods.hilrp.{module}"), entry)
        target = int(refs[name][0].argmax())
        first = attribute(models[name], x, target=target)
        with torch.no_grad():
            patched = models[name](x)
        out[name] = dict(attribute=attribute, target=target, ref=refs[name],
                         patched=patched, first=first)
        if name in bias_free:
            out[name]["bias_free"] = attribute(bias_free[name], x)

    for name, run in out.items():               # again, with every port's patches live
        run["again"] = run["attribute"](models[name], x, target=run["target"])
    return out


@pytest.mark.parametrize("name", PORTS)
def test_patches_leave_the_forward_pass_unchanged(runs, name):
    run = runs[name]
    scale = max(run["ref"].abs().max().item(), 1.0)
    assert (run["patched"] - run["ref"]).abs().max().item() / scale < TOL_FORWARD
    # and under the zennit composite (Gamma hooks, merged BatchNorm)
    vanilla_logit = run["ref"][0, run["target"]].item()
    assert abs(run["first"]["logit"] - vanilla_logit) / scale < TOL_FORWARD


@pytest.mark.parametrize("name", PORTS)
def test_map_is_finite_and_not_blank(runs, name):
    pixel_map = runs[name]["first"]["pixel_map"]
    assert tuple(pixel_map.shape) == (224, 224)
    assert torch.isfinite(pixel_map).all()
    assert pixel_map.abs().sum() > 0


@pytest.mark.parametrize("name", PORTS)
def test_conservation_trace_is_complete_and_finite(runs, name):
    stage_sums = runs[name]["first"]["stage_sums"]
    assert stage_sums, "port returned no conservation trace"
    assert all(torch.isfinite(torch.tensor(v)) for _, v in stage_sums)


@pytest.mark.parametrize("name", PORTS)
def test_map_is_unchanged_once_every_port_has_patched(runs, name):
    first, again = runs[name]["first"]["pixel_map"], runs[name]["again"]["pixel_map"]
    assert (again - first).abs().max() / first.abs().max() <= TOL_COEXIST


@pytest.mark.parametrize("name", CONSERVING)
def test_bias_free_swin_and_pvt_conserve_at_every_capture(runs, name):
    res = runs[name]["bias_free"]
    trace = dict(res["stage_sums"], pixels=res["pixel_map"].sum().item() / res["logit"])
    worst = max(trace, key=lambda k: abs(trace[k] - 1.0))
    assert abs(trace[worst] - 1.0) < TOL_CONSERVE, f"{worst}: sum(R)/logit = {trace[worst]:.6f}"

"""AttnLRP baseline method registered in the bench registry.

Two paths, matching the lineage experiments (scripts/lineage_naive_legs.py):

FLAT ViT (vit_*, deit_*, beit_*): the true published recipe. LXT efficient
backend with ATTN_MODE='attnlrp':
  - Softmax Taylor rule (Proposition 3.1 of Achtibat et al. 2024) via the plain
    softmax vjp (automatic when not detaching the attn matrix)
  - Uniform bilinear rule on attn@v via divide_gradient (Eq. 7)

HIERARCHICAL / HYBRID (swin, pvt, maxvit, mobilevit, efficientvit): the NAIVE
generic recipe, i.e. what a user gets today applying LXT's flat-transformer
recipe to an unsupported architecture: nn.GELU and nn.LayerNorm identity
patches plus zennit Gamma composites, and NOTHING architecture-specific.
Deliberately generous: it also gets the BN-merge canonizer and correct
preprocessing (generic zennit/bench hygiene). It does NOT get the HiLRP
contribution set (window/separable/linear attention rules, the norm-subclass
guard, activation identities beyond GELU). Same protocol as the n=1000
lineage/conservation runs.

This is the BASELINE from Achtibat et al. (2024), NOT HiLRP.

DEDICATED-RUN NOTE: Like hilrp, both paths apply class-level forward patches.
Do not mix with gradient-based baselines in the same process. Also do not mix
the flat-ViT path and the naive path in ONE process: attribute_vit patches the
timm norm subclasses, which would silently un-naive the hierarchical baseline.
Run them from separate configs (configs/attnlrp_grid_vit.yaml vs
configs/attnlrp_grid_hier.yaml).
"""

import torch
import torch.nn as nn

from xai_bench.registry import METHODS


def _is_flat_vit(model_name: str) -> bool:
    return any(model_name.startswith(p) for p in ("vit", "deit", "beit"))


_NAIVE_PATCHED = False


def _naive_patch_once():
    """LXT's generic recipe, class-level: GELU + nn.LayerNorm only."""
    global _NAIVE_PATCHED
    if _NAIVE_PATCHED:
        return
    from lxt.efficient.patches import patch_method, non_linear_forward, layer_norm_forward
    from lxt.efficient.zennit_patches import monkey_patch_zennit

    patch_method(non_linear_forward, nn.GELU, keep_original=True)
    patch_method(layer_norm_forward, nn.LayerNorm)
    monkey_patch_zennit()
    _NAIVE_PATCHED = True


def _bn_pairs(model):
    pairs = []
    for name, mod in model.named_modules():
        conv = getattr(mod, "conv", None)
        if not isinstance(conv, nn.Conv2d):
            continue
        for bn_attr in ("bn", "norm"):
            bn = getattr(mod, bn_attr, None)
            if isinstance(bn, nn.BatchNorm2d):
                pairs.append(([f"{name}.conv"], f"{name}.{bn_attr}"))
                break
    return pairs


def _attribute_naive(model, x, target=None, gamma: float = 0.25):
    """Generic naive LXT attribution (lineage protocol), Gradient x Input."""
    from zennit.canonizers import NamedMergeBatchNorm
    from zennit.composites import LayerMapComposite
    from zennit.rules import Gamma

    x = x.clone().detach().requires_grad_(True)
    comp = LayerMapComposite(
        [(nn.Conv2d, Gamma(gamma)), (nn.Linear, Gamma(gamma))],
        canonizers=[NamedMergeBatchNorm(_bn_pairs(model))],
    )
    with comp.context(model) as mod:
        logits = mod(x)
        t = int(logits[0].argmax()) if target is None else int(target)
        mod.zero_grad(set_to_none=True)
        logits[0, t].backward()
    return (x * x.grad)[0].sum(0).detach().cpu()


@METHODS.register("attnlrp")
def get_attnlrp(model_wrapper, model=None, gamma: float = 0.25, **kwargs):
    """AttnLRP attribution (Achtibat et al. 2024).

    Flat ViTs: LXT efficient backend, ATTN_MODE='attnlrp'.
    Hierarchical/hybrid: generic naive LXT recipe (lineage protocol).
    """
    from xai_bench.models.timm_models import InputStatsAdapter

    inner = model_wrapper.model
    adapter = None
    if isinstance(inner, InputStatsAdapter):
        adapter, inner = inner, inner.model

    model_name = model_wrapper.model_name

    if _is_flat_vit(model_name):
        from xai_bench.methods.hilrp.vit_lxt import attribute_vit

        def method(inputs: torch.Tensor, target: int) -> torch.Tensor:
            x = inputs if adapter is None else adapter._adapt(inputs)
            # attn_mode='attnlrp' selects the full AttnLRP rules (not CP-LRP)
            res = attribute_vit(inner, x, target=target, gamma=gamma, attn_mode="attnlrp")
            out = res["pixel_map"].detach().unsqueeze(0).to(inputs.device)
            inner.zero_grad(set_to_none=True)
            if inputs.is_cuda:
                torch.cuda.empty_cache()
            return out

        return method

    _naive_patch_once()

    def method(inputs: torch.Tensor, target: int) -> torch.Tensor:
        x = inputs if adapter is None else adapter._adapt(inputs)
        pm = _attribute_naive(inner, x, target=target, gamma=gamma)
        out = pm.unsqueeze(0).to(inputs.device)
        inner.zero_grad(set_to_none=True)
        if inputs.is_cuda:
            torch.cuda.empty_cache()
        return out

    return method

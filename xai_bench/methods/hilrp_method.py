"""HiLRP as a registered bench method, so the standard runner computes the full
quantus grid (faithfulness, localization, robustness, complexity) for it.

DEDICATED-RUN REQUIREMENT: the HiLRP backend applies CLASS-level forward
patches (LayerNorm, GELU, attention classes). Inside one process every other
method's GRADIENTS are altered by those patches (forward numerics identical,
gradient paths not: detached-std LN, CP attention). Therefore hilrp rows MUST
be produced in a run whose methods list contains ONLY hilrp (see
configs/hilrp_grid.yaml); never mix it with baselines in one process. The same
lesson produced the two-pass Shapley protocol.
"""
import torch

from xai_bench.registry import METHODS


def _dispatch(model_name):
    if model_name.startswith("swin"):
        from xai_bench.methods.hilrp.swin_lxt import attribute_swin
        return attribute_swin
    if model_name.startswith("mobilevit"):
        from xai_bench.methods.hilrp.mobilevit_lxt import attribute_mobilevit
        return attribute_mobilevit
    if model_name.startswith("efficientvit"):
        from xai_bench.methods.hilrp.efficientvit_lxt import attribute_efficientvit
        return attribute_efficientvit
    if model_name.startswith("pvt"):
        from xai_bench.methods.hilrp.pvt_lxt import attribute_pvt
        return attribute_pvt
    if any(model_name.startswith(p) for p in ("vit", "deit", "beit")):
        from xai_bench.methods.hilrp.vit_lxt import attribute_vit
        return attribute_vit
    raise ValueError(f"no HiLRP attributor for {model_name}")


@METHODS.register("hilrp")
def get_hilrp(model_wrapper, model=None, gamma=0.25, **kwargs):
    """Conservation-based relevance flow (HiLRP). Single backward pass.

    Handles the bench's InputStatsAdapter transparently: the adapter is an
    exact per-channel affine, so inputs are converted to the model's native
    stats before attribution and the map's spatial layout is unchanged.
    """
    from xai_bench.models.timm_models import InputStatsAdapter

    inner = model_wrapper.model
    adapter = None
    if isinstance(inner, InputStatsAdapter):
        adapter, inner = inner, inner.model

    attribute = _dispatch(model_wrapper.model_name)

    def method(inputs: torch.Tensor, target: int) -> torch.Tensor:
        x = inputs if adapter is None else adapter._adapt(inputs)
        res = attribute(inner, x, target=target, gamma=gamma)
        return res["pixel_map"].unsqueeze(0).to(inputs.device)

    return method

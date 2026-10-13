"""HiLRP as a registered bench method, so the standard runner computes the full
quantus grid (faithfulness, localization, robustness, complexity) for it.

DEDICATED-RUN REQUIREMENT: the HiLRP backend applies CLASS-level forward
patches (LayerNorm, GELU, attention classes). Inside one process every other
method's GRADIENTS are altered by those patches (forward numerics identical,
gradient paths not: detached-std LN, CP attention). Therefore hilrp rows MUST
be produced in a run whose methods list contains ONLY hilrp (see
configs/hilrp_grid.yaml); never mix it with baselines in one process. The same
lesson produced the two-pass Shapley protocol. The runner enforces this: see
xai_bench/isolation.py.
"""
import torch

from xai_bench.registry import METHODS


def _dispatch(model_name):
    """The HiLRP port for a bench model name. The table lives in the package
    (hilrp/api.py) so the bench and `hilrp.attribute` cannot drift apart."""
    from hilrp import attributor_for
    return attributor_for(model_name)


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
        # quantus calls the explain_func hundreds of times per image
        # (max_sensitivity samples, faithfulness perturbations). Each call opens
        # a composite context and a backward graph; free the CUDA cache between
        # calls or fragmentation eventually triggers "CUDA error: unknown error".
        x = inputs if adapter is None else adapter._adapt(inputs)
        res = attribute(inner, x, target=target, gamma=gamma)
        out = res["pixel_map"].detach().unsqueeze(0).to(inputs.device)
        inner.zero_grad(set_to_none=True)
        if inputs.is_cuda:
            torch.cuda.empty_cache()
        return out

    return method

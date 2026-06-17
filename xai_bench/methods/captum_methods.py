import torch
from captum.attr import IntegratedGradients, Saliency, NoiseTunnel

from xai_bench.registry import METHODS


class CaptumMethodWrapper:
    def __init__(self, model: torch.nn.Module, method_cls, **kwargs):
        self.model = model
        self.method = method_cls(model, **kwargs)

    def __call__(self, inputs: torch.Tensor, target: int, **kwargs) -> torch.Tensor:
        # Most Captum methods return an attribution tensor of the same shape as the input.
        # We usually want to return a 2D spatial map [H, W].
        attr = self.method.attribute(inputs, target=target, **kwargs)
        # Sum over color channels and return absolute value or ReLU?
        # Usually, sum over channels and absolute value is a standard fallback for IG/Saliency.
        attr = attr.sum(dim=1)  # [B, C, H, W] -> [B, H, W]
        return attr.abs()


@METHODS.register("saliency")
def get_saliency(model: torch.nn.Module, **kwargs):
    def method(inputs, target):
        wrapper = CaptumMethodWrapper(model, Saliency)
        # abs=True is standard for vanilla Saliency to match paper
        return wrapper(inputs, target, abs=True)
    return method


@METHODS.register("integrated_gradients")
def get_ig(model: torch.nn.Module, **kwargs):
    def method(inputs, target):
        wrapper = CaptumMethodWrapper(model, IntegratedGradients)
        return wrapper(inputs, target, n_steps=50)
    return method


@METHODS.register("smoothgrad")
def get_smoothgrad(model: torch.nn.Module, **kwargs):
    def method(inputs, target):
        saliency = Saliency(model)
        nt = NoiseTunnel(saliency)
        attr = nt.attribute(inputs, nt_type='smoothgrad', nt_samples=20, target=target, abs=True)
        return attr.sum(dim=1).abs()
    return method

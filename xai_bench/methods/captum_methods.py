import torch
# pyrefly: ignore [missing-import]
from captum.attr import IntegratedGradients, Saliency, NoiseTunnel, InputXGradient

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


@METHODS.register("input_x_gradient")
def get_input_x_gradient(model: torch.nn.Module, **kwargs):
    def method(inputs, target):
        wrapper = CaptumMethodWrapper(model, InputXGradient)
        return wrapper(inputs, target)
    return method


@METHODS.register("vargrad")
def get_vargrad(model: torch.nn.Module, **kwargs):
    def method(inputs, target):
        saliency = Saliency(model)
        nt = NoiseTunnel(saliency)
        attr = nt.attribute(inputs, nt_type='vargrad', nt_samples=20, target=target, abs=False)
        return attr.sum(dim=1).abs()
    return method


@METHODS.register("lrp")
def get_lrp(model: torch.nn.Module, **kwargs):
    from captum.attr import LRP
    def method(inputs, target):
        wrapper = CaptumMethodWrapper(model, LRP)
        return wrapper(inputs, target)
    return method


@METHODS.register("lime")
def get_lime(model: torch.nn.Module, **kwargs):
    from captum.attr import Lime
    def method(inputs, target):
        wrapper = CaptumMethodWrapper(model, Lime)
        return wrapper(inputs, target, n_samples=50) # Keep samples low for speed
    return method


@METHODS.register("gradient_shap")
def get_gradient_shap(model: torch.nn.Module, **kwargs):
    from captum.attr import GradientShap
    def method(inputs, target):
        wrapper = CaptumMethodWrapper(model, GradientShap)
        baselines = torch.zeros_like(inputs)
        return wrapper(inputs, target, baselines=baselines, n_samples=20)
    return method




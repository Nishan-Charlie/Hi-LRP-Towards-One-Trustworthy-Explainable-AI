"""Optional memory cap for batched gradient methods (NEUCOM revision).

SmoothGrad, VarGrad and GradientSHAP evaluate their 20 noisy copies of an image in
one batch. On MaxViT-S that batch overflows the 8 GB laptop GPU into shared system
memory and runs an order of magnitude slower. With the environment variable
XAI_GRAD_CHUNK=k set, captum's gradient function processes such batches k samples
at a time. Captum draws the noise before it calls the gradient function, and in
eval mode each sample's gradient depends only on that sample, so the attribution
is unchanged up to floating-point rounding. Off (a no-op) unless the variable is set.
"""
import os

import torch
import captum._utils.gradient as _gradient
import captum.attr._utils.attribution as _attribution

_ORIGINAL = _gradient.compute_gradients


def chunked_compute_gradients(forward_fn, inputs, target_ind=None, additional_forward_args=None):
    k = int(os.environ.get("XAI_GRAD_CHUNK", "0") or 0)
    tpl = inputs if isinstance(inputs, tuple) else (inputs,)
    n = tpl[0].shape[0]
    if k <= 0 or n <= k or additional_forward_args is not None:
        return _ORIGINAL(forward_fn, inputs, target_ind, additional_forward_args)
    parts = []
    for s in range(0, n, k):
        # captum calls this under torch.no_grad() (NoiseTunnel); the slices must be
        # taken with grad enabled so that they stay connected to the autograd graph
        with torch.enable_grad():
            sub = tuple(x[s:s + k] for x in tpl)
        t = target_ind
        if isinstance(t, torch.Tensor) and t.dim() > 0 and t.shape[0] == n:
            t = t[s:s + k]
        elif isinstance(t, (list, tuple)) and len(t) == n:
            t = t[s:s + k]
        parts.append(_ORIGINAL(forward_fn, sub, t, None))
    return tuple(torch.cat([p[i] for p in parts]) for i in range(len(tpl)))


def enable():
    """GradientAttribution.__init__ binds this module-level name to gradient_func."""
    _attribution.compute_gradients = chunked_compute_gradients


if int(os.environ.get("XAI_GRAD_CHUNK", "0") or 0) > 0:
    enable()

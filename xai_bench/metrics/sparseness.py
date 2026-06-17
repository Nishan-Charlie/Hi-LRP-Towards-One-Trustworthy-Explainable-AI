import torch
import numpy as np

from xai_bench.registry import METRICS

@METRICS.register("gini")
def gini_coefficient(attr: torch.Tensor, **kwargs) -> float:
    """Computes the Gini coefficient of the attribution map.
    Higher Gini = more sparse/concentrated.
    """
    attr_flat = attr.flatten().abs()
    attr_flat = torch.sort(attr_flat)[0]
    n = attr_flat.numel()
    
    if attr_flat.sum() == 0:
        return 0.0
        
    index = torch.arange(1, n + 1, device=attr.device, dtype=torch.float32)
    gini = (2 * torch.sum(index * attr_flat) / (n * torch.sum(attr_flat))) - (n + 1) / n
    return gini.item()

@METRICS.register("entropy")
def shannon_entropy(attr: torch.Tensor, **kwargs) -> float:
    """Computes Shannon entropy of the normalized attribution map.
    Lower entropy = more sparse.
    """
    attr_flat = attr.flatten().abs()
    attr_sum = attr_flat.sum()
    if attr_sum == 0:
        return 0.0
        
    p = attr_flat / attr_sum
    p = p[p > 0] # remove zeros for log
    
    entropy = -torch.sum(p * torch.log2(p))
    return entropy.item()

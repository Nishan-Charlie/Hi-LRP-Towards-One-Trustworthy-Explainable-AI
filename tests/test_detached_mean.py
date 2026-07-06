import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
import timm

from xai_bench.methods.hilrp.mobilevit_lxt import attribute_mobilevit, MOBILEVIT_PATCH_MAP

# We will redefine group_norm1_forward to detach the mean
def group_norm1_forward_detached(self, x):
    dims = tuple(range(1, x.dim()))
    mean = x.mean(dim=dims, keepdim=True)
    var = ((x - mean) ** 2).mean(dim=dims, keepdim=True)
    std = (var + self.eps).sqrt()
    # DETACH MEAN to prevent spatial smearing of relevance!
    y = (x - mean.detach()) / std.detach()
    if self.weight is not None:
        shape = (1, -1) + (1,) * (x.dim() - 2)
        y = y * self.weight.view(shape) + self.bias.view(shape)
    return y

# Update the patch map
from lxt.efficient.patches import patch_method
from timm.layers.norm import GroupNorm1
MOBILEVIT_PATCH_MAP[GroupNorm1] = partial(patch_method, group_norm1_forward_detached)

# Load model
model = timm.create_model("mobilevitv2_100", pretrained=True)
model.eval()

# Load image
img = Image.open("images/cat.jpg").convert("RGB")
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
x = transform(img).unsqueeze(0)

# Attribute
res = attribute_mobilevit(model, x, target=281, gamma=0.25, conv_gamma=0.1)

print("Stage sums with detached mean:")
for name, val in res["stage_sums"]:
    print(f"  {name}: {val:.4f}")

# Save the pixel map so we can look at it or analyze it
m = res["pixel_map"]
print(f"Detached mean map: min={m.min().item():.3e}, max={m.max().item():.3e}, mean={m.mean().item():.3e}")

import matplotlib.pyplot as plt
import numpy as np

# Plot it
m_np = m.numpy()
m_np = np.maximum(m_np, 0)
m_np = m_np / (m_np.max() + 1e-8)
plt.imshow(m_np, cmap='inferno')
plt.axis('off')
plt.savefig("cat_mobilevit_detached_mean.png", bbox_inches='tight')
print("Saved cat_mobilevit_detached_mean.png")

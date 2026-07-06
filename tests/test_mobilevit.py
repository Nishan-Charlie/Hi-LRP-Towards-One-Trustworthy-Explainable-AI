import sys
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
sys.path.insert(0, ".")
from xai_bench.methods.hilrp.mobilevit_lxt import attribute_mobilevit

device = "cuda" if torch.cuda.is_available() else "cpu"
model = timm.create_model("mobilevitv2_100", pretrained=True).eval().to(device)

config = resolve_data_config({}, model=model)
transform = create_transform(**config)
img_path = "images/cat.jpg"
pil_img = Image.open(img_path).convert('RGB')
x = transform(pil_img).unsqueeze(0).to(device)

res = attribute_mobilevit(model, x)

print("Target:", res["target"])
print("Logit:", res["logit"])
print("Stage sums:", res["stage_sums"])
pm = res["pixel_map"].numpy()
print("Pixel map max:", np.max(pm), "min:", np.min(pm))

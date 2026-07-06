import torch
import timm
from PIL import Image
from torchvision import transforms
from xai_bench.methods.hilrp.mobilevit_lxt import attribute_mobilevit

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

# Attribute with different gammas
for cg in [0.0, 0.1, 0.25]:
    res = attribute_mobilevit(model, x, target=281, gamma=0.25, conv_gamma=cg)
    m = res["pixel_map"]
    print(f"conv_gamma={cg}: min={m.min().item():.3e}, max={m.max().item():.3e}, mean={m.mean().item():.3e}, std={m.std().item():.3e}")
    # Also sort to see if there's a massive outlier
    top5 = torch.topk(m.flatten(), 5).values.tolist()
    print(f"  Top 5 vals: {[f'{v:.3e}' for v in top5]}")

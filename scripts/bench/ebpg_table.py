"""Format results/ebpg_results.json into a LaTeX EBPG table and a bbox-PG vs EBPG
comparison for the CAM methods (to show the de-saturation)."""
import json

R = json.load(open("results/ebpg_results.json"))
MODEL_ORDER = ["resnet50", "vit_base_patch16_224", "swin_base_patch4_window7_224",
               "pvt_v2_b2", "maxvit_small_tf_224", "mobilevitv2_100",
               "efficientvit_b1", "efficientvit_b2"]
COLS = ["RN-50", "ViT-B/16", "Swin-B", "PVT-v2", "MaxViT-S", "MobileViT", "EffViT-B1", "EffViT-B2"]
METHODS = [("saliency", "Saliency"), ("input_x_gradient", "Input$\\times$Grad"),
           ("smoothgrad", "SmoothGrad"), ("vargrad", "VarGrad"),
           ("grad_cam", "Grad-CAM"), ("grad_cam_plus_plus", "Grad-CAM++")]
# bbox-PG from tab:pg for the CAM rows (to show de-saturation)
BBOX_PG = {
    "grad_cam":          [0.98, 0.93, 1.00, 0.84, 1.00, 1.00, 0.70, 0.55],
    "grad_cam_plus_plus":[0.97, 0.49, 1.00, 0.88, 0.80, 1.00, 0.44, 0.56],
}


def val(model, method, key):
    r = R.get(f"{model}|{method}")
    return r[key] if r else float("nan")


print("=== EBPG table (energy-in-mask, random baseline = mean mask area 0.18) ===")
for mk, ml in METHODS:
    cells = " & ".join(f"{val(m, mk, 'ebpg'):.2f}" for m in MODEL_ORDER)
    print(f"    {ml} & {cells} \\\\")

print("\n=== strict peak-in-mask PG (dense silhouette) ===")
for mk, ml in METHODS:
    cells = " & ".join(f"{val(m, mk, 'pgd'):.2f}" for m in MODEL_ORDER)
    print(f"    {ml} & {cells} \\\\")

print("\n=== bbox-PG vs dense-EBPG for CAM (de-saturation) ===")
print(f"{'method/model':22s} " + " ".join(f"{c:>10s}" for c in COLS))
for mk, ml in [("grad_cam", "Grad-CAM"), ("grad_cam_plus_plus", "Grad-CAM++")]:
    bb = BBOX_PG[mk]
    eb = [val(m, mk, "ebpg") for m in MODEL_ORDER]
    print(f"{ml:22s} " + " ".join(f"{b:.2f}->{e:.2f}" for b, e in zip(bb, eb)))

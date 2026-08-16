import os
import sys
import glob
import numpy as np
import torch
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

DATASETS = {
    "imagenets": os.path.join("data", "ImageNetS", "cache_validation_*.pt"),
}

def load_cache(dataset="imagenets"):
    caches = glob.glob(DATASETS[dataset])
    path = max(caches, key=lambda p: int(p.rsplit("_", 1)[-1].split(".")[0]))
    return torch.load(path, map_location="cpu", weights_only=False), path

def pointing(pm, bboxes):
    h, w = pm.shape
    py, px = np.unravel_index(np.argmax(pm), pm.shape)
    return any(b[0] * w <= px <= b[2] * w and b[1] * h <= py <= b[3] * h for b in bboxes)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def run_classic_lrp(model_name, n, dataset="imagenets"):
    from xai_bench.registry import MODELS, METHODS
    data, path = load_cache(dataset)
    n = min(n, len(data))
    mw = MODELS.get(model_name)(); mw.model.eval().to(DEVICE)
    
    # Try to load lrp
    try:
        lrp = METHODS.get("lrp")(model_wrapper=mw, model=mw.model)
    except Exception as e:
        print(f"Failed to initialize LRP for {model_name}: {e}")
        return

    hits = np.zeros(n, dtype=bool)
    for i in range(n):
        x = data[i]["image"].unsqueeze(0).to(DEVICE)
        # Adapt stats if required
        if hasattr(mw.model, "_adapt"):
            x = mw.model._adapt(x)
            
        with torch.no_grad():
            t = int(mw.model(x).argmax())
            
        try:
            m = lrp(x, t); m = (m[0] if m.ndim == 3 else m).detach().cpu().numpy()
            hits[i] = pointing(m, data[i]["metadata"]["bboxes"])
        except Exception as e:
            print(f"Error on image {i}: {e}")
            break
            
        if (i + 1) % 10 == 0:
            print(f"  classic LRP {i+1}/{n}  running {hits[:i+1].mean():.3f}", flush=True)
            
    print(f"classic LRP {model_name} [{dataset}]: {hits.mean():.3f} (n={n})")

if __name__ == "__main__":
    for model in ["resnet50", "convnext_base"]:
        run_classic_lrp(model, 100)

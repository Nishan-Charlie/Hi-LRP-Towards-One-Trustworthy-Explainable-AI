import os
from typing import Callable, Optional, Tuple, Dict, Any

import torch
from torch.utils.data import Dataset
from torchvision import transforms

from xai_bench.registry import DATASETS

class ImageNetSDataset(Dataset):
    """
    ImageNet-S dataset loaded via Hugging Face `datasets`.
    """
    def __init__(self, split: str = "validation", transform: Optional[Callable] = None, num_samples: Optional[int] = None):
        # lazy import: HF `datasets` is only needed for streaming, not for the
        # cached .pt loader (imagenets_cached) -- keeps envs without it working
        from datasets import load_dataset
        self.transform = transform

        print(f"Loading ImageNet-S {split} split...")
        ds = load_dataset("braceletboy/imagenet-s", split=split, streaming=True)
        
        self.samples = []
        for i, item in enumerate(ds):
            if num_samples is not None and i >= num_samples:
                break
            self.samples.append(item)
            
    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, Dict[str, Any]]:
        item = self.samples[idx]
        img_pil = item['image']
        label = item['label']
        
        if img_pil.mode != "RGB":
            img_pil = img_pil.convert("RGB")
            
        metadata = {'img_name': f"sample_{idx}.jpg", 'bboxes': []}
        
        if 'mask' in item:
            import numpy as np
            from torchvision.transforms import InterpolationMode
            resize = transforms.Resize((224, 224), interpolation=InterpolationMode.NEAREST)
            mask_resized = np.array(resize(item['mask']))
            rows = np.any(mask_resized > 0, axis=1)
            cols = np.any(mask_resized > 0, axis=0)
            if np.any(rows):
                ymin, ymax = np.where(rows)[0][[0, -1]]
                xmin, xmax = np.where(cols)[0][[0, -1]]
                metadata['bboxes'] = [[
                    float(xmin) / 224.0, 
                    float(ymin) / 224.0, 
                    float(xmax) / 224.0, 
                    float(ymax) / 224.0
                ]]

        if self.transform:
            img = self.transform(img_pil)
        else:
            img = img_pil
            
        return img, label, metadata

class ImageNetSCached(Dataset):
    """Reads pre-materialised ImageNet-S samples from a local .pt cache so a long
    benchmark run never depends on (flaky) HuggingFace streaming and is fully
    restart-safe. Build the cache with `cache_imagenets.py`."""
    def __init__(self, cache_path: str, num_samples: Optional[int] = None):
        if not os.path.exists(cache_path):
            raise FileNotFoundError(
                f"Cache not found: {cache_path}. Build it first with cache_imagenets.py"
            )
        self.records = torch.load(cache_path, weights_only=False)
        if num_samples is not None:
            self.records = self.records[:num_samples]
        print(f"Loaded {len(self.records)} cached ImageNet-S samples from {cache_path}")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, Dict[str, Any]]:
        rec = self.records[idx]
        return rec["image"], rec["label"], rec["metadata"]


@DATASETS.register("imagenets_cached")
def get_imagenets_cached_dataset(**kwargs):
    cache_path = kwargs.get("cache_path", "data/ImageNetS/cache_validation_100.pt")
    num_samples = kwargs.get("num_samples", None)
    return ImageNetSCached(cache_path=cache_path, num_samples=num_samples)


@DATASETS.register("imagenets")
def get_imagenets_dataset(**kwargs):
    num_samples = kwargs.get('num_samples', 50)
    split = kwargs.get('split', 'validation')
    
    from torchvision import transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return ImageNetSDataset(split=split, transform=transform, num_samples=num_samples)

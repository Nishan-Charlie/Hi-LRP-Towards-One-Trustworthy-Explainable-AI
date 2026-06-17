import os
from typing import Callable, Optional, Tuple, Dict, Any
import xml.etree.ElementTree as ET

import torch
from torch.utils.data import Dataset
from torchvision.datasets.folder import default_loader

from xai_bench.registry import DATASETS


class ImageNetBBoxDataset(Dataset):
    """A standard ImageNet validation loader that also parses bounding boxes.
    
    Expected structure:
    root/
        ILSVRC2012_val_00000001.JPEG
        ...
    bboxes/
        val/
            ILSVRC2012_val_00000001.xml
            ...
    labels.txt
        # lines of form: ILSVRC2012_val_00000001.JPEG <class_id>
    """
    def __init__(self, root_dir: str, bbox_dir: Optional[str] = None, 
                 labels_file: Optional[str] = None, transform: Optional[Callable] = None):
        self.root_dir = root_dir
        self.bbox_dir = bbox_dir
        self.transform = transform
        
        self.samples = []
        if labels_file and os.path.exists(labels_file):
            with open(labels_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        img_name = parts[0]
                        label = int(parts[1])
                        self.samples.append((img_name, label))
        else:
            # dummy demo mode if no labels provided
            for file in os.listdir(root_dir):
                if file.endswith(".JPEG") or file.endswith(".jpg"):
                    self.samples.append((file, 0)) # Fake label 0

    def __len__(self) -> int:
        return len(self.samples)

    def _parse_bbox(self, xml_path: str, orig_width: int, orig_height: int) -> list:
        # Returns normalized bboxes [xmin, ymin, xmax, ymax] in [0, 1] range
        bboxes = []
        if not os.path.exists(xml_path):
            return bboxes
            
        tree = ET.parse(xml_path)
        root = tree.getroot()
        for obj in root.findall('object'):
            bndbox = obj.find('bndbox')
            if bndbox is not None:
                xmin = float(bndbox.find('xmin').text) / orig_width
                ymin = float(bndbox.find('ymin').text) / orig_height
                xmax = float(bndbox.find('xmax').text) / orig_width
                ymax = float(bndbox.find('ymax').text) / orig_height
                bboxes.append([xmin, ymin, xmax, ymax])
        return bboxes

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, Dict[str, Any]]:
        img_name, label = self.samples[idx]
        img_path = os.path.join(self.root_dir, img_name)
        img = default_loader(img_path)
        
        metadata = {'img_name': img_name, 'bboxes': []}
        
        if self.bbox_dir:
            base_name = os.path.splitext(img_name)[0]
            xml_path = os.path.join(self.bbox_dir, f"{base_name}.xml")
            metadata['bboxes'] = self._parse_bbox(xml_path, img.width, img.height)
            
        if self.transform:
            img = self.transform(img)
            
        return img, label, metadata


@DATASETS.register("demo")
def get_demo_dataset(**kwargs):
    # Uses local data/demo directory by default
    root = kwargs.get('root', 'data/demo/images')
    bbox_dir = kwargs.get('bbox_dir', 'data/demo/bboxes')
    labels = kwargs.get('labels_file', 'data/demo/labels.txt')
    
    # Needs transforms applied externally typically, but we can do a simple fallback
    from torchvision import transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    os.makedirs(root, exist_ok=True)
    os.makedirs(bbox_dir, exist_ok=True)
    
    # We will generate synthetic data in our test script later if this is empty
    return ImageNetBBoxDataset(root, bbox_dir, labels, transform=transform)

import torch
import torch.nn.functional as F
import numpy as np
from captum.attr import Occlusion

from xai_bench.registry import METHODS
from .captum_methods import CaptumMethodWrapper

@METHODS.register("occlusion")
def get_occlusion(model: torch.nn.Module, **kwargs):
    def method(inputs, target):
        wrapper = CaptumMethodWrapper(model, Occlusion)
        # Occlusion requires a sliding window shape.
        # For a 224x224 image, a common window is (3, 15, 15) or (3, 16, 16).
        # We assume input is [B, C, H, W]
        C, H, W = inputs.shape[-3:]
        window_shape = (C, min(H, 15), min(W, 15))
        strides = (C, min(H, 8), min(W, 8))
        return wrapper(inputs, target, sliding_window_shapes=window_shape, strides=strides)
    return method


class RISE:
    """Randomized Input Sampling for Explanation (RISE)"""
    def __init__(self, model, input_size=(224, 224), num_masks=2000, p1=0.5, s=8):
        self.model = model
        self.input_size = input_size
        self.num_masks = num_masks
        self.p1 = p1
        self.s = s
        self.masks = None
        
    def generate_masks(self, device):
        # Generate random downsampled masks
        # s is grid size (e.g. 8x8)
        H, W = self.input_size
        cell_size = np.ceil(np.array([H, W]) / self.s)
        up_size = (self.s + 1) * cell_size
        
        # Binary masks
        grid = (torch.rand(self.num_masks, 1, self.s, self.s, device=device) < self.p1).float()
        
        # Upsample using bilinear interpolation
        masks = F.interpolate(grid, size=(int(up_size[0]), int(up_size[1])), mode='bilinear', align_corners=False)
        
        # Random shifts
        shift_x = torch.randint(0, int(cell_size[1]), (self.num_masks,), device=device)
        shift_y = torch.randint(0, int(cell_size[0]), (self.num_masks,), device=device)
        
        # Crop to input size
        final_masks = torch.empty((self.num_masks, 1, H, W), device=device)
        for i in range(self.num_masks):
            final_masks[i] = masks[i, :, shift_y[i]:shift_y[i]+H, shift_x[i]:shift_x[i]+W]
            
        self.masks = final_masks
        
    def __call__(self, inputs: torch.Tensor, target: int, **kwargs):
        device = inputs.device
        if self.masks is None or self.masks.device != device:
            self.generate_masks(device)
            
        B, C, H, W = inputs.shape
        # Ensure single image for simple RISE implementation
        if B > 1:
            raise NotImplementedError("RISE currently only supports batch size 1")
            
        attr = torch.zeros((H, W), device=device)
        
        # Batched inference for masks to speed up
        batch_size = 128
        
        self.model.eval()
        with torch.no_grad():
            for i in range(0, self.num_masks, batch_size):
                mask_batch = self.masks[i:i+batch_size]
                masked_inputs = inputs * mask_batch
                
                logits = self.model(masked_inputs)
                probs = F.softmax(logits, dim=1)[:, target]
                
                # Multiply probability by mask and add
                attr += (mask_batch.squeeze(1) * probs.view(-1, 1, 1)).sum(dim=0)
                
        # Normalize
        attr = attr / (self.num_masks * self.p1)
        return attr.unsqueeze(0) # [1, H, W]


@METHODS.register("rise")
def get_rise(model: torch.nn.Module, **kwargs):
    def method(inputs, target):
        input_size = inputs.shape[-2:]
        # Use a reduced num_masks (e.g. 500) to keep execution reasonable for a benchmark
        # Real applications often use 2000-8000
        wrapper = RISE(model, input_size=input_size, num_masks=500)
        return wrapper(inputs, target)
    return method

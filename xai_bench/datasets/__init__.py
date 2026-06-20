"""Dataset loaders for benchmark execution.

Provides standard PyTorch datasets that return (image, target_class, metadata),
where metadata contains ground-truth localization information (like bounding boxes).
"""

from . import imagenet_val
from . import imagenet_s

# Benchmark Datasets

This directory stores datasets for the explainability benchmark.

## Expected Directory Structure

### Demo Dataset
```
data/demo/
├── images/
│   ├── sample1.jpg
│   └── sample2.jpg
├── bboxes/
│   ├── sample1.xml
│   └── sample2.xml
└── labels.txt
```

### ImageNet Validation (with bounding boxes)
```
data/imagenet/
├── val/
│   ├── ILSVRC2012_val_00000001.JPEG
│   └── ...
├── bboxes/
│   └── val/
│       ├── ILSVRC2012_val_00000001.xml
│       └── ...
└── val_labels.txt
```
To run on ImageNet, you need the standard `ILSVRC2012` validation set and the accompanying bounding box annotations.

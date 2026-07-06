import torch
import numpy as np
from xai_bench.config import RunConfig
from xai_bench.runner import BenchmarkRunner

config = RunConfig.from_yaml("configs/imagenets_quantitative.yaml")
config.num_images = 1
config.models = ["vit_base_patch16_224"]
config.methods = ["grad_cam"]
config.metrics = ["faithfulness_correlation"]
config.save_saliency = False
config.output_dir = "results"
config.run_name = "test"

# Run it directly but remove the try-except to see the error
runner = BenchmarkRunner(config)

from xai_bench.registry import MODELS, METHODS, METRICS, DATASETS
dataset_fn = DATASETS.get(runner.config.dataset)
dataset = dataset_fn(**runner.config.dataset_kwargs)
img, target, metadata = dataset[0]

from xai_bench.registry import MODELS, METHODS, METRICS

model_wrapper = MODELS.get("vit_base_patch16_224")()
model_wrapper.model.to("cuda").eval()

method_fn = METHODS.get("grad_cam")
method = method_fn(model_wrapper=model_wrapper, model=model_wrapper.model)

batch_img = img.unsqueeze(0).to("cuda")
attr = method(batch_img, target=target)

def explain_func(model, inputs, targets, **kwargs):
    device = next(model.parameters()).device
    inputs_t = torch.tensor(inputs, device=device, dtype=torch.float32)
    target_t = targets[0].item() if hasattr(targets, '__len__') else targets
    return method(inputs_t, target=target_t).cpu().numpy()

metric_fn = METRICS.get("faithfulness_correlation")
try:
    val = metric_fn(
        attr=attr[0], 
        metadata=metadata,
        model=model_wrapper.model,
        image=img.to("cuda"),
        target=target,
        explain_func=explain_func
    )
    print("Success:", val)
except Exception as e:
    import traceback
    traceback.print_exc()

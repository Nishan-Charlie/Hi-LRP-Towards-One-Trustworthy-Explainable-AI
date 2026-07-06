import torch
import traceback
from xai_bench.config import RunConfig
from xai_bench.runner import BenchmarkRunner
from xai_bench.registry import MODELS, METHODS, METRICS, DATASETS

config = RunConfig.from_yaml("configs/temp_vit.yaml")
runner = BenchmarkRunner(config)

dataset_fn = DATASETS.get(runner.config.dataset)
dataset = dataset_fn(**runner.config.dataset_kwargs)
img, target, metadata = dataset[0]

model_wrapper = MODELS.get("vit_base_patch16_224")()
model_wrapper.model.to("cuda").eval()

method_fn = METHODS.get("saliency")
method = method_fn(model_wrapper=model_wrapper, model=model_wrapper.model)

batch_img = img.unsqueeze(0).to("cuda")
attr = method(batch_img, target=target)

def explain_func(model, inputs, targets, **kwargs):
    device = next(model.parameters()).device
    inputs_t = torch.tensor(inputs, device=device, dtype=torch.float32)
    target_t = targets[0].item() if hasattr(targets, '__len__') else targets
    return method(inputs_t, target=target_t).cpu().numpy()

print(f"Testing metrics: {config.metrics}")
for metric_name in config.metrics:
    try:
        metric_fn = METRICS.get(metric_name)
        val = metric_fn(
            attr=attr[0], 
            metadata=metadata,
            model=model_wrapper.model,
            image=img.to("cuda"),
            target=target,
            explain_func=explain_func
        )
        print(f"Success {metric_name}: {val}")
    except Exception as e:
        print(f"FAILED {metric_name}: {e}")
        traceback.print_exc()

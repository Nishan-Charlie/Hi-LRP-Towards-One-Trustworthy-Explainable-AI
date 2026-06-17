import os
import json
import time
import datetime
from typing import Dict, Any

import torch
from tqdm import tqdm

from xai_bench.config import RunConfig
from xai_bench.registry import MODELS, METHODS, DATASETS, METRICS


class BenchmarkRunner:
    def __init__(self, config: RunConfig):
        self.config = config
        self.device = torch.device(config.resolved_device())
        os.makedirs(self.config.output_dir, exist_ok=True)
        
        if not self.config.run_name:
            self.config.run_name = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            
        self.run_dir = os.path.join(self.config.output_dir, self.config.run_name)
        os.makedirs(self.run_dir, exist_ok=True)
        
        # Save config
        with open(os.path.join(self.run_dir, "config.json"), "w") as f:
            json.dump(self.config.to_dict(), f, indent=2)

    def run(self):
        dataset_fn = DATASETS.get(self.config.dataset)
        dataset = dataset_fn(**self.config.dataset_kwargs)
        
        num_images = self.config.num_images if self.config.num_images else len(dataset)
        num_images = min(num_images, len(dataset))
        
        results = []

        for model_name in self.config.models:
            print(f"Loading model: {model_name}")
            model_fn = MODELS.get(model_name)
            model_wrapper = model_fn()
            model_wrapper.model.to(self.device)
            model_wrapper.model.eval()

            for method_name in self.config.methods:
                print(f"  Setting up method: {method_name}")
                if not model_wrapper.is_vit and method_name == "attention_rollout":
                    if self.config.limit_methods_to_supported:
                        print(f"    Skipping {method_name} for CNN model {model_name}")
                        continue
                        
                method_fn = METHODS.get(method_name)
                try:
                    method = method_fn(model_wrapper=model_wrapper, model=model_wrapper.model)
                except Exception as e:
                    print(f"    Failed to initialize {method_name}: {e}")
                    continue

                for i in tqdm(range(num_images), desc=f"    Evaluating {method_name}"):
                    img, target, metadata = dataset[i]
                    img = img.to(self.device)
                    
                    try:
                        # Some methods need a batch dim
                        if img.dim() == 3:
                            batch_img = img.unsqueeze(0)
                        else:
                            batch_img = img
                            
                        # Generate attribution
                        t0 = time.time()
                        attr = method(batch_img, target=target)
                        attr_time = time.time() - t0
                        
                        # Calculate metrics
                        row = {
                            "model": model_name,
                            "method": method_name,
                            "image_idx": i,
                            "image_name": metadata.get('img_name', str(i)),
                            "target": target,
                            "time_ms": attr_time * 1000,
                        }
                        
                        for metric_name in self.config.metrics:
                            metric_fn = METRICS.get(metric_name)
                            # Pass extra kwargs for faithfulness
                            val = metric_fn(
                                attr=attr[0], 
                                metadata=metadata,
                                model=model_wrapper.model,
                                image=img,
                                target=target
                            )
                            row[metric_name] = val
                            
                        results.append(row)
                        
                        # Optionally save saliency map
                        if self.config.save_saliency:
                            save_path = os.path.join(self.run_dir, f"{model_name}_{method_name}_{i}.pt")
                            torch.save(attr[0].cpu(), save_path)
                            
                    except Exception as e:
                        print(f"    Error on image {i} with {method_name}: {e}")
                        
            # Free model to save VRAM
            del model_wrapper
            torch.cuda.empty_cache()
            
        # Save results
        with open(os.path.join(self.run_dir, "results.json"), "w") as f:
            json.dump(results, f, indent=2)
            
        print(f"Run complete. Results saved to {self.run_dir}")

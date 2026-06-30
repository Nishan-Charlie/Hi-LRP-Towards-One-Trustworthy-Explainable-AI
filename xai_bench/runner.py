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

    def _write_csv(self, results):
        """Write cumulative results to CSV files. Called after every model so a
        long run produces usable output incrementally and survives a crash."""
        if not results:
            return
        try:
            import pandas as pd
        except ImportError:
            print("pandas not available; skipping CSV export.")
            return

        df = pd.DataFrame(results)
        # Full per-image results
        df.to_csv(os.path.join(self.run_dir, "results.csv"), index=False)

        # Aggregated summary: mean +/- std per (model, method) across images
        metric_cols = [m for m in self.config.metrics if m in df.columns]
        if metric_cols:
            agg = df.groupby(["model", "method"])[metric_cols].agg(["mean", "std", "count"])
            agg.columns = [f"{metric}_{stat}" for metric, stat in agg.columns]
            agg = agg.reset_index()
            agg.to_csv(os.path.join(self.run_dir, "summary.csv"), index=False)
        print(f"  CSV updated: {len(df)} rows -> {os.path.join(self.run_dir, 'results.csv')}")

    def run(self):
        dataset_fn = DATASETS.get(self.config.dataset)
        dataset = dataset_fn(**self.config.dataset_kwargs)
        
        num_images = self.config.num_images if self.config.num_images else len(dataset)
        num_images = min(num_images, len(dataset))
        
        results = []
        completed_evals = set()
        results_file = os.path.join(self.run_dir, "results.json")
        if os.path.exists(results_file):
            print(f"Resuming from existing results.json in {self.run_dir}")
            try:
                with open(results_file, "r") as f:
                    results = json.load(f)
                for r in results:
                    completed_evals.add((r["model"], r["method"], r["image_idx"]))
                print(f"Loaded {len(results)} existing results.")
            except Exception as e:
                print(f"Failed to load existing results.json: {e}")

        for model_name in self.config.models:
            print(f"Loading model: {model_name}")
            model_fn = MODELS.get(model_name)
            model_wrapper = model_fn()
            model_wrapper.model.to(self.device)
            model_wrapper.model.eval()

            for method_name in self.config.methods:
                print(f"  Setting up method: {method_name}")
                supports_attention = getattr(model_wrapper, "supports_attention", model_wrapper.is_vit)
                if not supports_attention and method_name in ("attention_rollout", "attention_gradient"):
                    if self.config.limit_methods_to_supported:
                        print(f"    Skipping {method_name}: {model_name} has no global CLS-token attention")
                        continue
                        
                method_fn = METHODS.get(method_name)
                try:
                    method = method_fn(model_wrapper=model_wrapper, model=model_wrapper.model)
                except Exception as e:
                    print(f"    Failed to initialize {method_name}: {e}")
                    continue

                for i in tqdm(range(num_images), desc=f"    Evaluating {method_name}"):
                    if (model_name, method_name, i) in completed_evals:
                        continue
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
                        if isinstance(attr, tuple):
                            attr = tuple(a.detach() if isinstance(a, torch.Tensor) else a for a in attr)
                        elif isinstance(attr, torch.Tensor):
                            attr = attr.detach()
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
                        
                        def explain_func(model, inputs, targets, **kwargs):
                            import torch
                            device = next(model.parameters()).device
                            inputs_t = torch.tensor(inputs, device=device, dtype=torch.float32)
                            target_t = targets[0].item() if hasattr(targets, '__len__') else targets
                            # detach: CAM/attention methods return grad-requiring tensors
                            return method(inputs_t, target=target_t).detach().cpu().numpy()

                        for metric_name in self.config.metrics:
                            metric_fn = METRICS.get(metric_name)
                            # Pass extra kwargs for faithfulness + per-metric overrides
                            val = metric_fn(
                                attr=attr[0],
                                metadata=metadata,
                                model=model_wrapper.model,
                                image=img,
                                target=target,
                                explain_func=explain_func,
                                metric_kwargs=self.config.metric_kwargs.get(metric_name, {}),
                            )
                            import math
                            try:
                                val = float(val)
                                if math.isnan(val):
                                    val = None
                            except:
                                pass
                            row[metric_name] = val
                            
                        results.append(row)
                        
                        # Save results iteratively to avoid data loss on crash
                        with open(os.path.join(self.run_dir, "results.json"), "w") as f:
                            json.dump(results, f, indent=2)                        
                        # Optionally save saliency map
                        if self.config.save_saliency:
                            save_path = os.path.join(self.run_dir, f"{model_name}_{method_name}_{i}.pt")
                            torch.save(attr[0].cpu(), save_path)
                            
                    except Exception as e:
                        print(f"    Error on image {i} with {method_name}: {e}")
                        
            # Update CSV after every model finishes (explicit requirement)
            print(f"Finished model: {model_name}. Writing CSV...")
            self._write_csv(results)

            # Free model to save VRAM
            del model_wrapper
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Save results
        with open(os.path.join(self.run_dir, "results.json"), "w") as f:
            json.dump(results, f, indent=2)
        self._write_csv(results)

        print(f"Run complete. Results saved to {self.run_dir}")

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Run XAI benchmark")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    args = parser.parse_args()
    
    config = RunConfig.from_yaml(args.config)
    runner = BenchmarkRunner(config)
    runner.run()

if __name__ == "__main__":
    main()

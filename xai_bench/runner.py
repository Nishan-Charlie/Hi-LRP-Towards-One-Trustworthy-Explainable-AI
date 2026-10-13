import os
import json
import time
import zlib
import datetime
from typing import Dict, Any

import torch
from tqdm import tqdm

from xai_bench import isolation
from xai_bench.config import RunConfig
from xai_bench.registry import MODELS, METHODS, DATASETS, METRICS
from xai_bench.utils.seed import set_seed


def evaluation_seed(seed: int, model_name: str, method_name: str, image_idx: int) -> int:
    """The RNG seed for one (seed, model, method, image) evaluation.

    Derived per evaluation, not drawn from one stream, so a row's randomness does
    not depend on which rows ran before it: resuming a crashed run, reordering
    `methods`, or dropping a model leaves every remaining row unchanged. crc32
    rather than hash() because str hashes are salted per interpreter.
    """
    return zlib.crc32(f"{seed}|{model_name}|{method_name}|{image_idx}".encode())


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
        # Before any model loads: refuse a run whose gradients this interpreter
        # can no longer compute honestly (class-level LRP patches already live).
        isolation.check_process(self.config.methods, self.config.models)

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
                # Rows written before seeds were recorded carry no "seed" field.
                # Keep them (count them under the first seed) rather than
                # silently recomputing a finished run, but say what they are.
                unseeded = sum("seed" not in r for r in results)
                for r in results:
                    seed = r.get("seed", self.config.seeds[0])
                    completed_evals.add((r["model"], r["method"], r["image_idx"], seed))
                print(f"Loaded {len(results)} existing results.")
                if unseeded:
                    print(f"WARNING: {unseeded} of them predate seed recording and were "
                          f"NOT produced under a fixed seed; they are kept as-is. "
                          f"Use a new run_name for a fully seeded run.")
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

                evals = [(seed, i) for seed in self.config.seeds for i in range(num_images)]
                for seed, i in tqdm(evals, desc=f"    Evaluating {method_name}"):
                    if (model_name, method_name, i, seed) in completed_evals:
                        continue
                    # Covers everything stochastic in this evaluation: dataset
                    # augmentation, the attribution, and every metric after it.
                    set_seed(evaluation_seed(seed, model_name, method_name, i))
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
                            "seed": seed,
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
                            # single-seed runs keep the original file name
                            tag = f"_s{seed}" if len(self.config.seeds) > 1 else ""
                            save_path = os.path.join(self.run_dir, f"{model_name}_{method_name}_{i}{tag}.pt")
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

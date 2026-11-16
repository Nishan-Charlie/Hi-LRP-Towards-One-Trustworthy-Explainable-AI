import datetime
import json
import math
import os
import time
import zlib

import torch
from tqdm import tqdm

from xai_bench import isolation, scoring
from xai_bench.config import RunConfig
from xai_bench.registry import DATASETS, METHODS, METRICS, MODELS
from xai_bench.utils.seed import set_seed


def evaluation_seed(seed: int, model_name: str, method_name: str, image_idx: int) -> int:
    """The RNG seed for one (seed, model, method, image) evaluation.

    Derived per evaluation, not drawn from one stream, so a row's randomness does
    not depend on which rows ran before it: resuming a crashed run, reordering
    `methods`, or dropping a model leaves every remaining row unchanged. crc32
    rather than hash() because str hashes are salted per interpreter.
    """
    return zlib.crc32(f"{seed}|{model_name}|{method_name}|{image_idx}".encode())


def _sync(device: torch.device) -> None:
    """Wait for queued GPU work, so a timed window covers the whole computation."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _error_text(e: Exception) -> str:
    return f"{type(e).__name__}: {e}"[:300]


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
        long run produces usable output incrementally and survives a crash.

        summary.csv, per (model, method):
          n_evaluations       rows attempted (images x seeds)
          n_failed            rows whose attribution raised (kept in results.json
                              with an "error" field; retried when the run resumes)
          n_constant_maps     rows whose attribution map is constant
          <metric>_mean/median/std/count
                              over every successful row, constant maps and
                              undefined values scored as in xai_bench.scoring (no
                              image is dropped); the paper reports
                              scoring.cell_statistic(metric) (the median for MS,
                              whose failures score +inf, the mean otherwise)
          <metric>_mean_defined / <metric>_undefined
                              the raw mean over the rows where the metric is
                              defined, and the number of rows where it is not
          time_ms_*, peak_gpu_mb_*   attribution cost, GPU-synchronized
        results.json keeps the raw values (None where undefined)."""
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

        keys = ["model", "method"]
        if "error" not in df.columns:
            df["error"] = None
        failed = df["error"].notna()
        ok = df[~failed]
        summary = df.groupby(keys).size().rename("n_evaluations").to_frame()
        summary["n_failed"] = failed.groupby([df[k] for k in keys]).sum().astype(int)

        if "degenerate_map" in ok.columns and len(ok):
            summary["n_constant_maps"] = ok["degenerate_map"].eq(True) \
                .groupby([ok[k] for k in keys]).sum().astype(int)
        metric_cols = [m for m in self.config.metrics if m in df.columns]
        if metric_cols and len(ok):
            raw = ok[keys + metric_cols].copy()
            for m in metric_cols:
                raw[m] = pd.to_numeric(raw[m], errors="coerce").replace([math.inf, -math.inf], math.nan)
            scored = scoring.score(ok, metric_cols, group=keys)
            agg = scored.groupby(keys)[metric_cols].agg(["mean", "median", "std", "count"])
            agg.columns = [f"{metric}_{stat}" for metric, stat in agg.columns]
            graw = raw.groupby(keys)
            for m in metric_cols:
                agg[f"{m}_mean_defined"] = graw[m].mean()
                agg[f"{m}_undefined"] = graw[m].apply(lambda s: int(s.isna().sum()))
            summary = summary.join(agg, how="left")
        for cost in ("time_ms", "peak_gpu_mb"):
            if cost in ok.columns and len(ok):
                t = pd.to_numeric(ok[cost], errors="coerce").groupby([ok[k] for k in keys])
                summary = summary.join(t.agg(["mean", "median", "std"]).add_prefix(f"{cost}_"), how="left")
        summary.reset_index().to_csv(os.path.join(self.run_dir, "summary.csv"), index=False)
        print(f"  CSV updated: {len(df)} rows ({int(failed.sum())} failed) -> "
              f"{os.path.join(self.run_dir, 'results.csv')}")

    def _write_failures(self, results, init_failures):
        """failures.json: methods that could not be set up, and every failed evaluation."""
        evals = [{k: r.get(k) for k in ("model", "method", "seed", "image_idx", "error")}
                 for r in results if r.get("error")]
        with open(os.path.join(self.run_dir, "failures.json"), "w") as f:
            json.dump({"method_init": init_failures, "evaluations": evals}, f, indent=2)

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
                with open(results_file) as f:
                    results = json.load(f)
                # Failed evaluations are recorded, not dropped; a resumed run retries
                # them (a CUDA OOM or a crash mid-image is usually transient).
                retry = [r for r in results if r.get("error")]
                if retry:
                    results = [r for r in results if not r.get("error")]
                    print(f"Retrying {len(retry)} evaluations that failed in the previous attempt.")
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

        init_failures = {}
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
                    init_failures[f"{model_name}/{method_name}"] = _error_text(e)
                    print(f"    FAILED to initialize {method_name}: {_error_text(e)} "
                          f"(recorded in failures.json)")
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
                    row = {
                        "model": model_name,
                        "method": method_name,
                        "image_idx": i,
                        "image_name": metadata.get('img_name', str(i)),
                        "seed": seed,
                        "target": target,
                    }

                    # Attribution. A failure is recorded as a row with an "error"
                    # field (metrics None), so it is counted in summary.csv and
                    # failures.json instead of disappearing from the averages.
                    try:
                        batch_img = img.unsqueeze(0) if img.dim() == 3 else img
                        if self.device.type == "cuda":
                            torch.cuda.reset_peak_memory_stats(self.device)
                        _sync(self.device)
                        t0 = time.perf_counter()
                        attr = method(batch_img, target=target)
                        if isinstance(attr, tuple):
                            attr = tuple(a.detach() if isinstance(a, torch.Tensor) else a for a in attr)
                        elif isinstance(attr, torch.Tensor):
                            attr = attr.detach()
                        _sync(self.device)
                        row["time_ms"] = (time.perf_counter() - t0) * 1000
                        if self.device.type == "cuda":
                            row["peak_gpu_mb"] = torch.cuda.max_memory_allocated(self.device) / 2 ** 20
                    except Exception as e:
                        row["error"] = _error_text(e)
                        row["time_ms"] = None
                        for metric_name in self.config.metrics:
                            row[metric_name] = None
                        print(f"    FAILED image {i} (seed {seed}) with {method_name}: {row['error']}")
                        results.append(row)
                        with open(results_file, "w") as f:
                            json.dump(results, f, indent=2)
                        if self.device.type == "cuda":
                            torch.cuda.empty_cache()
                        continue

                    # Quantus calls this with keyword arguments; the current method is
                    # bound explicitly so the closure cannot see a later loop value
                    def explain_func(model, inputs, targets, _method=method, **kwargs):
                        device = next(model.parameters()).device
                        inputs_t = torch.tensor(inputs, device=device, dtype=torch.float32)
                        target_t = targets[0].item() if hasattr(targets, '__len__') else targets
                        # detach: CAM/attention methods return grad-requiring tensors
                        return _method(inputs_t, target=target_t).detach().cpu().numpy()

                    # A constant map carries no information and every metric is
                    # undefined on it; it is flagged here and scored in summary.csv
                    # (xai_bench.scoring), not computed and not dropped.
                    row["degenerate_map"] = scoring.is_constant_map(attr[0])

                    # Metrics, each on its own: one failing metric no longer drops the
                    # others. A failed metric is None and its reason is kept.
                    metric_errors = {}
                    for metric_name in self.config.metrics:
                        if row["degenerate_map"]:
                            row[metric_name] = None
                            continue
                        metric_fn = METRICS.get(metric_name)
                        try:
                            val = metric_fn(
                                attr=attr[0],
                                metadata=metadata,
                                model=model_wrapper.model,
                                image=img,
                                target=target,
                                explain_func=explain_func,
                                metric_kwargs=self.config.metric_kwargs.get(metric_name, {}),
                            )
                        except Exception as e:
                            metric_errors[metric_name] = _error_text(e)
                            val = None
                        try:
                            val = float(val)
                            if not math.isfinite(val):
                                val = None   # undefined: counted and scored in summary.csv
                        except (TypeError, ValueError):
                            pass
                        row[metric_name] = val
                    if metric_errors:
                        row["metric_errors"] = metric_errors
                        print(f"    Metric failure on image {i} with {method_name}: {metric_errors}")

                    results.append(row)
                    # Save results iteratively to avoid data loss on crash
                    with open(results_file, "w") as f:
                        json.dump(results, f, indent=2)
                    # Optionally save saliency map
                    if self.config.save_saliency:
                        # single-seed runs keep the original file name
                        tag = f"_s{seed}" if len(self.config.seeds) > 1 else ""
                        save_path = os.path.join(self.run_dir, f"{model_name}_{method_name}_{i}{tag}.pt")
                        torch.save(attr[0].cpu(), save_path)

            # Update CSV after every model finishes (explicit requirement)
            print(f"Finished model: {model_name}. Writing CSV...")
            self._write_csv(results)
            self._write_failures(results, init_failures)

            # Free model to save VRAM
            del model_wrapper
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Save results
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2)
        self._write_csv(results)
        self._write_failures(results, init_failures)

        n_failed = sum(1 for r in results if r.get("error"))
        print(f"Run complete. Results saved to {self.run_dir}")
        if n_failed or init_failures:
            print(f"WARNING: {n_failed} failed evaluations and {len(init_failures)} methods that "
                  f"could not be set up; see failures.json. Rerun to retry the failed evaluations.")

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

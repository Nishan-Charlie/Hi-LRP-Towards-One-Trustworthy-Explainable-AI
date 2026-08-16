import argparse
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from xai_bench.config import RunConfig
from xai_bench.runner import BenchmarkRunner

def main():
    parser = argparse.ArgumentParser(description="Run XAI Benchmark")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to config yaml")
    args = parser.parse_args()

    print(f"Loading config from {args.config}")
    config = RunConfig.from_yaml(args.config)
    
    runner = BenchmarkRunner(config)
    runner.run()

if __name__ == "__main__":
    main()

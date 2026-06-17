import argparse
import json
import os
import pandas as pd

def main():
    parser = argparse.ArgumentParser(description="Analyze XAI Benchmark Results")
    parser.add_argument("results_dir", type=str, help="Path to the run directory containing results.json")
    args = parser.parse_args()

    results_file = os.path.join(args.results_dir, "results.json")
    if not os.path.exists(results_file):
        print(f"Error: Could not find {results_file}")
        return

    with open(results_file, 'r') as f:
        data = json.load(f)

    if not data:
        print("Results file is empty.")
        return

    df = pd.DataFrame(data)
    print(f"Loaded {len(df)} evaluation records.")
    
    # Calculate means grouped by model and method
    metrics = [col for col in df.columns if col not in ['model', 'method', 'image_idx', 'image_name', 'target', 'time_ms']]
    
    if not metrics:
        print("No metrics found in results.")
        return
        
    print("\n--- Mean Results ---")
    grouped = df.groupby(['model', 'method'])[metrics + ['time_ms']].mean().reset_index()
    print(grouped.to_string(index=False))
    
    # Save to CSV
    csv_path = os.path.join(args.results_dir, "summary.csv")
    grouped.to_csv(csv_path, index=False)
    print(f"\nSummary saved to {csv_path}")

if __name__ == "__main__":
    main()

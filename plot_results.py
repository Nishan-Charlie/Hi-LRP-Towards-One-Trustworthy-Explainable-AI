import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# Set research-based plot aesthetics
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'ggplot')
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 12

# Load data
with open("results/quant_vit_all/results.json", "r") as f:
    data = json.load(f)
df = pd.DataFrame(data)

# Calculate mean and std for each method and metric
metrics = ["faithfulness_correlation", "auc", "pointing_game"]
metric_names = {"faithfulness_correlation": "Faithfulness (Corr)", "auc": "Localization AUC", "pointing_game": "Pointing Game"}

summary_mean = df.groupby("method")[metrics].mean()
summary_std = df.groupby("method")[metrics].std()

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Use a colormap
cmap = plt.get_cmap("Set2")

for i, metric in enumerate(metrics):
    ax = axes[i]
    means = summary_mean[metric]
    stds = summary_std[metric]
    
    x = np.arange(len(means))
    colors = [cmap(i % cmap.N) for i in range(len(means))]
    
    bars = ax.bar(x, means, yerr=stds, capsize=5, color=colors, edgecolor="black", alpha=0.8)
    
    ax.set_title(metric_names[metric], fontweight="bold")
    ax.set_xticks(x)
    # Format labels cleanly
    labels = [m.replace('_', ' ').title() for m in means.index]
    ax.set_xticklabels(labels, rotation=45, ha='right')
    
    if i == 0:
        ax.set_ylabel("Score", fontweight="bold")
    
    # Add values on top of bars
    for bar in bars:
        yval = bar.get_height()
        offset = 0.02 * ax.get_ylim()[1]
        ax.text(bar.get_x() + bar.get_width()/2, 
                yval + offset if yval > 0 else yval - offset, 
                f"{yval:.3f}", 
                ha='center', va='bottom' if yval > 0 else 'top', 
                fontsize=11)

plt.tight_layout()

# Save the plot
output_path = "results/quant_vit_all/metrics_plot.png"
os.makedirs(os.path.dirname(output_path), exist_ok=True)
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"Plot saved to {output_path}")

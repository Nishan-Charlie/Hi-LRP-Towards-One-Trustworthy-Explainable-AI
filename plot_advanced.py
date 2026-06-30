import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from math import pi

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'ggplot')
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 12

def format_name(name):
    return name.replace('_', ' ').title().replace('Cam', 'CAM').replace('Vit', 'ViT').replace('Resnet', 'ResNet')

def load_data(path):
    with open(path, "r") as f:
        data = json.load(f)
    df = pd.DataFrame(data)
    df['method_formatted'] = df['method'].apply(format_name)
    df['model_formatted'] = df['model'].apply(format_name)
    return df

def plot_grouped_bar(df, metric, title, ylabel, output_path):
    grouped = df.groupby(['method_formatted', 'model_formatted'])[metric].mean().unstack()
    
    # Sort methods by average performance across models
    grouped = grouped.loc[grouped.mean(axis=1).sort_values(ascending=False).index]
    
    fig, ax = plt.subplots(figsize=(14, 6))
    grouped.plot(kind='bar', ax=ax, colormap='Set2', edgecolor='black', alpha=0.8)
    
    ax.set_title(title, fontweight="bold")
    ax.set_ylabel(ylabel, fontweight="bold")
    ax.set_xlabel("")
    plt.xticks(rotation=45, ha='right')
    plt.legend(title='Model Architecture')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

def plot_scatter_tradeoff(df, x_metric, y_metric, title, xlabel, ylabel, output_path, log_x=False):
    means = df.groupby(['method_formatted', 'model_formatted'])[[x_metric, y_metric]].mean().reset_index()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    models = means['model_formatted'].unique()
    colors = plt.get_cmap("tab10")(np.linspace(0, 1, len(models)))
    
    for idx, model in enumerate(models):
        subset = means[means['model_formatted'] == model]
        ax.scatter(subset[x_metric], subset[y_metric], label=model, color=colors[idx], s=100, alpha=0.7, edgecolors='black')
        
        # Annotate top 5 methods for each model to avoid clutter
        top_subset = subset.nlargest(5, y_metric)
        for _, row in top_subset.iterrows():
            ax.annotate(row['method_formatted'], (row[x_metric], row[y_metric]), 
                        xytext=(5, 5), textcoords='offset points', fontsize=8, alpha=0.8)

    if log_x:
        ax.set_xscale('log')
        
    ax.set_title(title, fontweight="bold")
    ax.set_xlabel(xlabel, fontweight="bold")
    ax.set_ylabel(ylabel, fontweight="bold")
    plt.legend(title='Model Architecture', loc='best')
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

def plot_heatmap(df, model_name, metrics, metric_labels, output_path):
    subset = df[df['model'] == model_name]
    if subset.empty: return
    
    means = subset.groupby('method_formatted')[metrics].mean()
    
    # Normalize metrics to 0-1 for heatmap comparison (column-wise)
    normalized = (means - means.min()) / (means.max() - means.min() + 1e-8)
    
    fig, ax = plt.subplots(figsize=(8, 8))
    cax = ax.imshow(normalized, cmap='viridis', aspect='auto')
    
    ax.set_xticks(np.arange(len(metrics)))
    ax.set_yticks(np.arange(len(means.index)))
    ax.set_xticklabels(metric_labels, rotation=45, ha='right')
    ax.set_yticklabels(means.index)
    
    # Add values text
    for i in range(len(means.index)):
        for j in range(len(metrics)):
            val = means.iloc[i, j]
            color = 'black' if normalized.iloc[i, j] > 0.5 else 'white'
            ax.text(j, i, f'{val:.2f}', ha='center', va='center', color=color, fontsize=10)
            
    fig.colorbar(cax, ax=ax, label='Normalized Score (Higher is relatively better)')
    ax.set_title(f"Method Performance on {format_name(model_name)}", fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

def plot_radar(df, model_name, methods, metrics, metric_labels, output_path):
    subset = df[df['model'] == model_name]
    if subset.empty: return
    
    means = subset.groupby('method_formatted')[metrics].mean()
    
    # Only keep the specified methods if they exist
    valid_methods = [m for m in methods if m in means.index]
    if not valid_methods: return
    
    # Radar chart setup
    N = len(metrics)
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    ax.set_theta_offset(pi / 2)
    ax.set_theta_direction(-1)
    
    plt.xticks(angles[:-1], metric_labels)
    ax.set_rlabel_position(0)
    
    # Normalize globally for the radar to make sense
    global_max = df[metrics].max()
    
    colors = plt.get_cmap("tab10")(np.linspace(0, 1, len(valid_methods)))
    
    for idx, method in enumerate(valid_methods):
        values = means.loc[method].values.flatten().tolist()
        # Normalize
        values = [v / m if m > 0 else 0 for v, m in zip(values, global_max)]
        values += values[:1]
        
        ax.plot(angles, values, linewidth=2, linestyle='solid', label=method, color=colors[idx])
        ax.fill(angles, values, color=colors[idx], alpha=0.1)
        
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    ax.set_title(f"Radar Comparison on {format_name(model_name)}", fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def main():
    data_path = "results/full_benchmark_100/results.json"
    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found.")
        return
        
    output_dir = "results/full_benchmark_100/plots_advanced"
    os.makedirs(output_dir, exist_ok=True)
    
    df = load_data(data_path)
    
    print("Generating Grouped Bar Charts...")
    plot_grouped_bar(df, 'pointing_game', 'Pointing Game (Localization) by Architecture', 'Pointing Game Score', os.path.join(output_dir, 'grouped_pointing_game.png'))
    plot_grouped_bar(df, 'faithfulness_correlation', 'Faithfulness Correlation by Architecture', 'Correlation Score', os.path.join(output_dir, 'grouped_faithfulness.png'))
    
    print("Generating Scatter Trade-off Plots...")
    # Using sparseness or faithfulness as fidelity, time_ms as cost
    plot_scatter_tradeoff(df, 'time_ms', 'faithfulness_correlation', 'Fidelity vs Cost Trade-off', 'Time (ms) - Log Scale', 'Faithfulness Correlation', os.path.join(output_dir, 'scatter_fidelity_vs_cost.png'), log_x=True)
    # Using pointing_game as localization, max_sensitivity as robustness (note: lower max_sensitivity is better robustness, so let's invert it or just plot it)
    plot_scatter_tradeoff(df, 'pointing_game', 'max_sensitivity', 'Localization vs Sensitivity (Robustness)', 'Pointing Game (Higher = Better Localized)', 'Max Sensitivity (Lower = More Robust)', os.path.join(output_dir, 'scatter_localization_vs_robustness.png'))

    metrics = ['pointing_game', 'faithfulness_correlation', 'sparseness', 'max_sensitivity']
    metric_labels = ['Pointing Game', 'Faithfulness', 'Sparseness', 'Sensitivity']

    print("Generating Heatmaps...")
    for model in df['model'].unique():
        plot_heatmap(df, model, metrics, metric_labels, os.path.join(output_dir, f'heatmap_{model}.png'))

    print("Generating Radar Charts...")
    top_methods = ['Grad CAM', 'Attention Rollout', 'Rise', 'Integrated Gradients'] # Formatted names
    for model in df['model'].unique():
        plot_radar(df, model, top_methods, metrics, metric_labels, os.path.join(output_dir, f'radar_{model}.png'))
        
    print(f"All advanced plots saved to {output_dir}")

if __name__ == "__main__":
    main()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from math import pi

out_dir = "results/plots_publication"
os.makedirs(out_dir, exist_ok=True)

# 1. Load Data
df_summary = pd.read_csv("results/combined_summary.csv")
df_res1 = pd.read_csv("results/full_benchmark_100/results.csv")
df_res2 = pd.read_csv("results/vit_suite_10/results.csv")
df_res = pd.concat([df_res1, df_res2], ignore_index=True)

df_time = df_res.groupby(['model', 'method'])['time_ms'].mean().reset_index()
df = pd.merge(df_summary, df_time, on=['model', 'method'], how='left')

df = df[df['model'] != 'convnext_base']

# Setup Styling for Publication (Times New Roman, Black Axes)
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "font.size": 16,
    "axes.labelsize": 18,
    "axes.titlesize": 20,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 14,
    "axes.edgecolor": "black",
    "axes.labelcolor": "black",
    "xtick.color": "black",
    "ytick.color": "black",
    "text.color": "black",
    "axes.linewidth": 1.5,
    "lines.linewidth": 2.5,
    "figure.figsize": (10, 6),
    "axes.spines.top": False,
    "axes.spines.right": False
})

pretty_models = {
    'resnet50': 'ResNet-50',
    'vit_base_patch16_224': 'ViT-B',
    'deit_base_patch16_224': 'DeiT-B',
    'swin_base_patch4_window7_224': 'Swin-B',
    'maxvit_small_tf_224': 'MaxViT-S',
    'mobilevitv2_100': 'MobileViT v2',
    'efficientvit_b1': 'EffViT-B1',
    'efficientvit_b2': 'EffViT-B2'
}

pretty_methods = {
    'grad_cam': 'Grad-CAM',
    'grad_cam_plus_plus': 'Grad-CAM++',
    'gradient_shap': 'GradSHAP',
    'input_x_gradient': 'Inp x Grad',
    'integrated_gradients': 'IntGrad',
    'lime': 'LIME',
    'occlusion': 'Occlusion',
    'rise': 'RISE',
    'saliency': 'Saliency',
    'smoothgrad': 'SmoothGrad',
    'vargrad': 'VarGrad',
    'attention_gradient': 'AttnGrad',
    'attention_rollout': 'Rollout'
}

df['model_pretty'] = df['model'].map(pretty_models)
df['method_pretty'] = df['method'].map(pretty_methods)

colors = sns.color_palette("colorblind", n_colors=13).as_hex()
color_map = {m: c for m, c in zip(pretty_methods.values(), colors)}
marker_map = {m: mkr for m, mkr in zip(pretty_methods.values(), ['o','s','^','v','D','p','*','h','X','d','P','>','<'])}

# ---------------------------------------------------------
# Plot 1: Rankings Do Not Transfer
# ---------------------------------------------------------
ordered_models = ['ResNet-50', 'ViT-B', 'Swin-B', 'MaxViT-S', 'MobileViT v2', 'EffViT-B2']
df_p1 = df[df['model_pretty'].isin(ordered_models)].copy()
df_p1['model_pretty'] = pd.Categorical(df_p1['model_pretty'], categories=ordered_models, ordered=True)
df_p1 = df_p1.sort_values('model_pretty')

plt.figure(figsize=(10, 6))
methods_to_plot = ['Grad-CAM', 'Grad-CAM++', 'IntGrad', 'VarGrad', 'LIME']
for m in methods_to_plot:
    subset = df_p1[df_p1['method_pretty'] == m]
    if not subset.empty:
        plt.plot(subset['model_pretty'], subset['pointing_game_mean'], 
                 label=m, color=color_map[m], marker=marker_map[m], markersize=10, zorder=3)

plt.ylabel("Localization (Pointing Game) ↑")
plt.ylim(0, 1.05)
plt.legend(title="Method", bbox_to_anchor=(1.05, 1), loc='upper left', frameon=False)
plt.grid(axis='y', linestyle='--', alpha=0.6, zorder=0)
plt.tight_layout()
plt.savefig(f"{out_dir}/1_localization_transfer.png", dpi=300)
plt.savefig(f"{out_dir}/1_localization_transfer.pdf")
plt.close()

# ---------------------------------------------------------
# Plot 2: Cost vs Localization on ViT-B
# ---------------------------------------------------------
plt.figure(figsize=(8, 6))
df_vit = df[df['model_pretty'] == 'ViT-B'].copy()

for i, row in df_vit.iterrows():
    m = row['method_pretty']
    plt.scatter(row['time_ms'], row['pointing_game_mean'], 
                color=color_map[m], marker=marker_map[m], s=150, zorder=3, label=m)

annot_methods = ['Grad-CAM', 'Occlusion', 'LIME', 'Rollout', 'IntGrad']
for i, row in df_vit.iterrows():
    if row['method_pretty'] in annot_methods:
        plt.annotate(row['method_pretty'], (row['time_ms']*1.2, row['pointing_game_mean']), 
                     fontsize=12, zorder=4)

plt.xscale('log')
plt.xlabel("Cost (Execution Time ms) [Log Scale] →")
plt.ylabel("Fidelity (Pointing Game) ↑")
plt.grid(linestyle='--', alpha=0.6, zorder=0)
plt.tight_layout()
plt.savefig(f"{out_dir}/2_cost_fidelity_vitb.png", dpi=300)
plt.savefig(f"{out_dir}/2_cost_fidelity_vitb.pdf")
plt.close()

# ---------------------------------------------------------
# Plot 3: Multi-metric Radar Chart (Spider Plot) for ViT-B
# ---------------------------------------------------------
def make_spider(df_arch, title, filename):
    comp_methods = ['Grad-CAM', 'IntGrad', 'Rollout', 'LIME']
    d = df_arch[df_arch['method_pretty'].isin(comp_methods)].copy()
    
    max_ms = d['max_sensitivity_mean'].max() if not d['max_sensitivity_mean'].isna().all() else 1
    min_ms = d['max_sensitivity_mean'].min() if not d['max_sensitivity_mean'].isna().all() else 0
    if max_ms != min_ms:
        d['Rob (1-MS)'] = 1 - ((d['max_sensitivity_mean'] - min_ms) / (max_ms - min_ms))
    else:
        d['Rob (1-MS)'] = 0.5
        
    metrics = ['pointing_game_mean', 'faithfulness_estimate_mean', 'sparseness_mean', 'Rob (1-MS)']
    labels = ['Loc (PG)', 'Faith (FE)', 'Sparse (SP)', 'Rob (1-MS)']
    
    N = len(labels)
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]
    
    plt.rcParams.update({"axes.spines.top": True, "axes.spines.right": True})
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    ax.set_theta_offset(pi / 2)
    ax.set_theta_direction(-1)
    
    plt.xticks(angles[:-1], labels)
    ax.set_rlabel_position(0)
    plt.yticks([0.25, 0.5, 0.75, 1.0], ["0.25","0.5","0.75","1.0"], color="grey", size=10)
    plt.ylim(0, 1.05)
    
    for i, row in d.iterrows():
        m = row['method_pretty']
        values = []
        for met in metrics:
            val = row[met]
            if pd.isna(val): val = 0
            if met == 'pointing_game_mean': val = val / 1.0
            if met == 'faithfulness_estimate_mean': val = max(0, val) / 0.5
            if met == 'sparseness_mean': val = val / 1.0
            values.append(min(1.0, max(0.0, val)))
            
        values += values[:1]
        ax.plot(angles, values, linewidth=2.5, linestyle='solid', label=m, color=color_map[m])
        ax.fill(angles, values, color=color_map[m], alpha=0.1)
        
    plt.title(title, size=22, y=1.1)
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), frameon=False)
    plt.tight_layout()
    plt.savefig(f"{out_dir}/{filename}.png", dpi=300)
    plt.savefig(f"{out_dir}/{filename}.pdf")
    plt.close()
    plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False})

make_spider(df_vit, "Method Trade-offs (ViT-B)", "3_radar_vitb")

# ---------------------------------------------------------
# Plot 4: APF vs Standard FC 
# ---------------------------------------------------------
df_apf = df.dropna(subset=['apf_mean']).copy()
if not df_apf.empty:
    plt.figure(figsize=(10, 6))
    apf_agg = df_apf.groupby('method_pretty')[['faithfulness_correlation_mean', 'apf_mean']].mean().reset_index()
    apf_agg = apf_agg.sort_values('apf_mean', ascending=False)
    
    x = np.arange(len(apf_agg))
    width = 0.35
    
    plt.bar(x - width/2, apf_agg['faithfulness_correlation_mean'], width, label='Standard FC', color='gray')
    plt.bar(x + width/2, apf_agg['apf_mean'], width, label='APF', color='black')
    
    plt.ylabel("Faithfulness Score ↑")
    plt.title("Attention-Preserving Faithfulness vs. Standard")
    plt.xticks(x, apf_agg['method_pretty'], rotation=45, ha='right')
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(f"{out_dir}/4_apf_vs_fc.png", dpi=300)
    plt.savefig(f"{out_dir}/4_apf_vs_fc.pdf")
    plt.close()
else:
    print("No APF data available to plot yet.")

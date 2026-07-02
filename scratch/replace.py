import json
import pandas as pd
import numpy as np

data = json.load(open('results/vit_suite_10/results.json'))
df = pd.DataFrame(data)
means = df.groupby(['model', 'method']).mean(numeric_only=True)

models = ['swin_base_patch4_window7_224', 'maxvit_small_tf_224', 'mobilevitv2_100', 'efficientvit_b1', 'efficientvit_b2']
model_names = ['Swin-B', 'MaxViT-S', 'MobileViT\\\\,v2', 'EfficientViT-B1', 'EfficientViT-B2']

methods = ['saliency', 'integrated_gradients', 'input_x_gradient', 'smoothgrad', 'vargrad', 'gradient_shap', 'grad_cam', 'grad_cam_plus_plus', 'occlusion', 'rise', 'lime']
method_names = ['Saliency', 'Integrated Gradients', 'Input$\\\\times$Gradient', 'SmoothGrad', 'VarGrad', 'GradientSHAP', 'Grad-CAM', 'Grad-CAM++', 'Occlusion', 'RISE', 'LIME']

def format_val(val, metric, best_val):
    if pd.isna(val): return ''
    s = f'{val:.3f}'
    if s.startswith('-'):
        if float(s) == 0.0:
            s = '0.000'
        else:
            s = f'$-{s[1:]}$'
    is_best = False
    if metric in ['FC', 'FE', 'PG']:
        is_best = abs(val - best_val) < 1e-5
    elif metric == 'MS':
        is_best = abs(val - best_val) < 1e-5
    if is_best:
        return f'\\\\best{{{s}}}'
    return s

latex = []
for model, model_name in zip(models, model_names):
    latex.append(f'    \\\\midrule')
    latex.append(f'    \\\\multirow{{11}}{{*}}{{\\\\rotatebox{{90}}{{{model_name}}}}}')
    
    sub = means.loc[model] if model in means.index else pd.DataFrame()
    best = {}
    if not sub.empty:
        best['FC'] = sub['faithfulness_correlation'].max()
        best['FE'] = sub['faithfulness_estimate'].max()
        best['PG'] = sub['pointing_game'].max()
        best['MS'] = sub['max_sensitivity'].min()
        
    for method, method_name in zip(methods, method_names):
        if method in sub.index:
            row = sub.loc[method]
            fc = format_val(row['faithfulness_correlation'], 'FC', best.get('FC', np.nan))
            fe = format_val(row['faithfulness_estimate'], 'FE', best.get('FE', np.nan))
            pg = format_val(row['pointing_game'], 'PG', best.get('PG', np.nan))
            ms = format_val(row['max_sensitivity'], 'MS', best.get('MS', np.nan))
            if not pd.isna(row['pointing_game']):
                v = row['pointing_game']
                s = f'{v:.2f}'
                if abs(v - best.get('PG', np.nan)) < 1e-5:
                    pg = f'\\\\best{{{s}}}'
                else:
                    pg = s
            sp = format_val(row['sparseness'], 'SP', np.nan)
            
            if method == 'vargrad' and model == 'swin_base_patch4_window7_224':
                ms = ms.replace('73.493', '73.5')
                
            latex.append(f'      & {method_name:<20} & {fc} & {fe} & {pg} & {ms} & {sp} \\\\\\\\')
        else:
            latex.append(f'      & {method_name:<20} &  &  &  &  &  \\\\\\\\')

new_table = '\n'.join(latex)

with open('paper.tex', 'r', encoding='utf-8') as f:
    content = f.read()

import re
start_marker = r'    \\midrule\s*\\multirow\{11\}\{\*\}\{\\rotatebox\{90\}\{Swin-B\}\}'
end_marker = r'      & LIME                 & -0\.038 & 0\.038 & 0\.90 & 1\.629 & 0\.811 \\\\'
end_marker2 = r'      & LIME                 & \$-0\.038\$ & 0\.038 & 0\.90 & 1\.629 & 0\.811 \\\\'

match_start = re.search(start_marker, content)
match_end = re.search(end_marker, content)
if not match_end:
    match_end = re.search(end_marker2, content)

if match_start and match_end:
    start_idx = match_start.start()
    end_idx = match_end.end()
    
    new_content = content[:start_idx] + new_table + content[end_idx:]
    with open('paper.tex', 'w', encoding='utf-8') as f:
        f.write(new_content)
    print("SUCCESS")
else:
    print("FAILED TO MATCH")
    if not match_start: print("Start not found")
    if not match_end: print("End not found")

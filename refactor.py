import json
import os

def load_notebook(path):
    with open(path, 'r') as f:
        return json.load(f)

def save_notebook(nb, path):
    with open(path, 'w') as f:
        json.dump(nb, f, indent=1)

nb = load_notebook('final_draft.ipynb')

demo_cells = []
app_cells = []

# --- demo_utils.py ---
imports_src = ''.join(nb['cells'][12]['source'])
core_code_src = ''.join(nb['cells'][14]['source'])
with open('demo_utils.py', 'w') as f:
    f.write(imports_src + '\n\n' + core_code_src)

# --- Common header for both notebooks ---
header_md = {
    'cell_type': 'markdown',
    'metadata': {},
    'source': [
        '# Calibrated judge evaluation: mean and lower-tail CVaR\n',
        '**Note**: This codebase implements tests originally motivated by *Causal Judge Evaluation*.\n'
    ]
}

imports_setup_cell = {
    'cell_type': 'code',
    'execution_count': None,
    'metadata': {},
    'outputs': [],
    'source': [
        'from demo_utils import *\n',
        'import pandas as pd\n',
        'import numpy as np\n',
        'import matplotlib.pyplot as plt\n',
        'from tqdm.auto import tqdm\n',
        '\n',
        '# Configure plot aesthetics\n',
        'plt.rcParams.update({\n',
        '    "font.size": 12,\n',
        '    "axes.titlesize": 14,\n',
        '    "axes.labelsize": 12,\n',
        '    "grid.alpha": 0.3,\n',
        '    "figure.figsize": (8, 5),\n',
        '})\n'
    ]
}

# Config processing
config_md_demo = nb['cells'][11].copy()
config_md_demo['source'] = [line.replace('100', '30') for line in config_md_demo['source']]

config_md_app = nb['cells'][11].copy()
config_md_app['source'] = [line.replace('30', '100') for line in config_md_app['source']]

config_code_cell = nb['cells'][12].copy()
clean_config = []
for line in config_code_cell['source']:
    if 'import ' in line or 'from ' in line:
        pass 
    else:
        clean_config.append(line)

base_config_str = ''.join(clean_config)

# Demo uses 30
demo_config_str = base_config_str.replace('REPLICATIONS = 100', 'REPLICATIONS = 30').replace('AUDIT_REPLICATIONS = 100', 'AUDIT_REPLICATIONS = 30')
# App uses 100
app_config_str = base_config_str.replace('REPLICATIONS = 30', 'REPLICATIONS = 100').replace('AUDIT_REPLICATIONS = 30', 'AUDIT_REPLICATIONS = 100')

# Append override for demo config
demo_config_code_cell = config_code_cell.copy()
demo_config_code_cell['source'] = [demo_config_str + "\n\n# DEMO OVERRIDE: Only run core scenarios for narrative flow\nSCENARIOS = {k: SCENARIOS[k] for k in ['clear_tail_tradeoff', 'no_tradeoff']}\n"]

# --- demo.ipynb ---
demo_cells.append(header_md)
demo_cells.extend(nb['cells'][5:11])
demo_cells.append(config_md_demo)
demo_cells.append(imports_setup_cell)
demo_cells.append(demo_config_code_cell)
demo_cells.extend(nb['cells'][15:26]) # Truth and MC estimation + interpretations

# Cross policy transport fix
cross_policy_code = nb['cells'][33].copy()
source = ''.join(cross_policy_code['source'])
source = source.replace("t_cvar = truth_store[sc_name]['B']['true_q_alpha']", "t_cvar = np.quantile(y_b_audit, ALPHA)")
cross_policy_code['source'] = [source]
demo_cells.append(nb['cells'][32])
demo_cells.append(cross_policy_code)
demo_cells.extend(nb['cells'][34:40])

# Covariate rescue fix
cov_md = nb['cells'][44].copy()
cov_md['source'].append("\n> **Note:** The 2D covariate 'rescue' uses a polynomial ridge regression, compared to the 1D isotonic baseline. While an apples-to-oranges estimator shift, it illustrates the conceptual power of covariates in rescuing transport failures.\n")
demo_cells.extend(nb['cells'][40:44]) 
demo_cells.append(cov_md)
demo_cells.extend(nb['cells'][45:48])

# Final takeaways fix for demo.ipynb
takeaways_code = nb['cells'][65].copy()
src = ''.join(takeaways_code['source'])
src = src.replace("asub = audit_all_df[audit_all_df['scenario'] == sc_name]", "c_res = cross_policy_results[sc_name]")
src = src.replace("stable_rej = float(asub[asub['variant'] == 'stable']['reject'].mean())", "mean_p = float(c_res['transport_p_mean'])")
src = src.replace("fooled_rej = float(asub[asub['variant'] == 'fooled_judge']['reject'].mean())", "cvar_p = float(c_res['transport_p_cvar'])")
src = src.replace("f'fooled_judge rejects {fooled_rej:.0%}. '", "f'Mean p-value: {mean_p:.3f}. '")
src = src.replace("f'stable rejects {stable_rej:.0%} (nominal is 5%).'", "f'CVaR p-value: {cvar_p:.3f}.'")
src = src.replace("'Stable over-rejects due to uncorrected first-stage uncertainty.'", "'Requires a reliable pilot/audit slice.'")
takeaways_code['source'] = [src]

demo_cells.append(nb['cells'][64])
demo_cells.append(takeaways_code)
demo_cells.extend(nb['cells'][66:68])

# Clean stale notes
for cell in demo_cells:
    if cell['cell_type'] == 'markdown':
        src = ''.join(cell['source'])
        src = src.replace("Counterfactual Judge Evaluation", "Causal Judge Evaluation")
        src = src.replace("knife_edge stable rejection is 3.3%", "knife_edge stable rejection is 13.3%")
        cell['source'] = [src]

demo_nb = nb.copy()
demo_nb['cells'] = demo_cells
save_notebook(demo_nb, 'demo.ipynb')

# --- appendix_diagnostics.ipynb ---
app_config_code_cell = config_code_cell.copy()
app_config_code_cell['source'] = [app_config_str]

app_cells.append(header_md)
app_cells.append(config_md_app) # Use the 100 replication text block
app_cells.append(imports_setup_cell)
app_cells.append(app_config_code_cell)
app_cells.extend(nb['cells'][15:17]) # Data prep

app_cells.extend(nb['cells'][40:44]) # 3 lever robustness
app_cells.extend(nb['cells'][48:64]) # within policy + sensitivity

# Clean markdown text in appendix
for cell in app_cells:
    if cell['cell_type'] == 'markdown':
        src = ''.join(cell['source'])
        src = src.replace("Counterfactual Judge Evaluation", "Causal Judge Evaluation")
        src = src.replace("knife_edge stable rejection is 3.3%", "knife_edge stable rejection is 13.3%")
        cell['source'] = [src]

app_nb = nb.copy()
app_nb['cells'] = app_cells
save_notebook(app_nb, 'appendix_diagnostics.ipynb')
print("Notebooks written.")

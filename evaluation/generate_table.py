import pandas as pd
from pathlib import Path

ds_to_eval = ['grazer', 'jsrt'][0]
is_jsrt = ds_to_eval == 'jsrt'
print(f'Evaluating {ds_to_eval} dataset')
csv_path = Path(f'evaluation/csv_files/{ds_to_eval}')
# concatenate csv files
df = pd.concat([pd.read_csv(f) for f in csv_path.glob('*.csv')], ignore_index=True)

# rename metric
df['metric'] = df['metric'].replace({'tre': 'TRE', 'avg_surf_dist': 'ASD', 'dice': 'DSC'})

# fuse values to mean and std
df['value'] = '$' + df['value_mean'].round(1).astype(str) + '_{\pm' + df['value_std'].round(1).astype(str) + '}$'
df = df.drop(['value_mean', 'value_std'], axis=1)

df = df.pivot_table(index=['anatomy', 'metric'], columns='Method', values='value', aggfunc='first').reset_index()

# reorder
# Define the order of the categories
if is_jsrt:
    anatomy_order = ['lungs', 'heart', 'clavicles', 'average']
metric_order = ['DSC', 'ASD', 'TRE']

# Convert the 'anatomy' and 'metric' columns to categorical
if is_jsrt:
    df['anatomy'] = pd.Categorical(df['anatomy'], categories=anatomy_order, ordered=True)
df['metric'] = pd.Categorical(df['metric'], categories=metric_order, ordered=True)

# Sort the DataFrame by 'anatomy' and 'metric'
df = df.sort_values(['anatomy', 'metric'])

# Reset the index
df = df.reset_index(drop=True)

df = df.set_index(['anatomy', 'metric'])

# reorder columns
if is_jsrt:
    df = df[['ShapeFormer', 'Heatmap Regression', 'HeatRegSeg', 'cartesian', 'cartesian_sparse', 'nnUNet']]
else:
    df = df[['Heatmap Regression', 'Heatmap Regression 0.25', 'HeatRegSeg_1', 'uv']]

print(df.to_string())
print(df.to_latex())

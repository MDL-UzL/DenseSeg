import pandas as pd
from pathlib import Path

csv_path = Path('evaluation/csv_files')
# concatenate csv files
df = pd.concat([pd.read_csv(f) for f in csv_path.glob('*.csv')], ignore_index=True)

# rename metric
df['metric'] = df['metric'].replace({'tre': 'TRE', 'avg_surf_dist': 'ASD', 'dice': 'DSC'})

# fuse values to mean and std
df['value'] = df['value_mean'].round(1).astype(str) + '±' + df['value_std'].round(1).astype(str)
df = df.drop(['value_mean', 'value_std'], axis=1)

df = df.pivot_table(index=['anatomy', 'metric'], columns='Method', values='value', aggfunc='first').reset_index()

# reorder
# Define the order of the categories
anatomy_order = ['lungs', 'heart', 'clavicles', 'average']
metric_order = ['DSC', 'ASD', 'TRE']

# Convert the 'anatomy' and 'metric' columns to categorical
df['anatomy'] = pd.Categorical(df['anatomy'], categories=anatomy_order, ordered=True)
df['metric'] = pd.Categorical(df['metric'], categories=metric_order, ordered=True)

# Sort the DataFrame by 'anatomy' and 'metric'
df = df.sort_values(['anatomy', 'metric'])

# Reset the index
df = df.reset_index(drop=True)

df = df.set_index(['anatomy', 'metric'])

# reorder columns
df = df[['ShapeFormer', 'Heatmap Regression', 'cartesian', 'cartesian_sparse', 'polar', 'nnUNet']]

print(df)

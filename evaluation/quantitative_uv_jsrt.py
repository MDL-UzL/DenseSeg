import pandas as pd
import torch
from clearml import InputModel
from kornia.geometry import normalize_pixel_coordinates
from monai.metrics import DiceMetric
from torch.nn import functional as F
from tqdm import tqdm

from dataset.jsrt_dataset import JSRTDatasetUV
from models.uv_unet import UVUNet
from utils import convert_list_of_uv_to_coordinates
from clearml_ids import jsrt_model_ids

uv_method = 'cartesian'
print(f'Evaluating model with uv method {uv_method}')
ds = JSRTDatasetUV('test', uv_method.split('_')[0])
cl_model = InputModel(jsrt_model_ids[uv_method])
model = UVUNet.load(cl_model.get_weights(), 'cpu').eval()

dsc_metric = DiceMetric(include_background=True, reduction='none', num_classes=ds.N_CLASSES)
df = pd.DataFrame(columns=['file', 'anatomy', 'metric', 'value'])

with torch.inference_mode():
    for i, (img, lm, dist_map, seg_mask, uv_map) in enumerate(tqdm(ds, desc='Evaluating', unit='img')):
        seg_hat, uv_hat = model.predict(img.unsqueeze(0), mask_uv=True)
        # seg_hat, uv_hat = seg_mask.unsqueeze(0), uv_map.unsqueeze(0)

        # DICE
        dsc_value = dsc_metric(seg_hat, seg_mask.unsqueeze(0)).squeeze(0)

        # TRE
        uv_values = list(ds.get_anatomical_structure_uv_values().values())
        lm_hat = convert_list_of_uv_to_coordinates(uv_hat, uv_values, 'linear', k=5)
        lm_hat = torch.cat(lm_hat, dim=1).squeeze(0)
        tre = torch.linalg.vector_norm(lm - lm_hat, dim=1, ord=2)
        tre *= ds.PIXEL_RESOLUTION_MM

        # AVG SURF DIST
        C, H, W = dist_map.shape
        N = lm.shape[0]
        lm_hat_norm = normalize_pixel_coordinates(lm_hat, H, W)
        dist_values = F.grid_sample(dist_map.unsqueeze(0), lm_hat_norm.view(1, 1, N, 2), align_corners=True).squeeze()
        avg_surf_dist = dist_values.abs() * ds.PIXEL_RESOLUTION_MM

        for anat_idx, (anatomy, (start_idx, end_idx)) in enumerate(ds.get_anatomical_structure_index().items()):
            df = pd.concat([df, pd.DataFrame(
                {'file': i, 'anatomy': anatomy, 'metric': 'dice', 'value': dsc_value[anat_idx].item() * 100}
                , index=[0])], ignore_index=True)

            df = pd.concat([df, pd.DataFrame(
                {'file': i, 'anatomy': anatomy, 'metric': 'tre', 'value': tre[start_idx:end_idx].mean().item()}
                , index=[0])], ignore_index=True)

            df = pd.concat([df, pd.DataFrame(
                {'file': i, 'anatomy': anatomy, 'metric': 'avg_surf_dist',
                 'value': avg_surf_dist[anat_idx, start_idx:end_idx].mean().item()}
                , index=[0])], ignore_index=True)

        #break
# cluster anatomy
df['anatomy'] = df['anatomy'].replace({'left_lung': 'lungs', 'right_lung': 'lungs',
                                       'heart': 'heart', 'left_clavicle': 'clavicles', 'right_clavicle': 'clavicles'})

# filter to ASD
df_files = df.loc[df['metric'] == 'avg_surf_dist']
df_files = df_files.drop(['anatomy', 'metric'], axis=1)
# find best, median, worst file
df_files = df_files.groupby('file').mean().reset_index()
df_files = df_files.sort_values('value', ascending=True)
best_file = df_files.iloc[0]['file']
median_file = df_files.iloc[len(df_files) // 2]['file']
worst_file = df_files.iloc[-1]['file']

print(f'Best file: {best_file} with ASD {df_files.iloc[0]["value"]}')
print(f'Median file: {median_file} with ASD {df_files.iloc[len(df_files) // 2]["value"]}')
print(f'Worst file: {worst_file} with ASD {df_files.iloc[-1]["value"]}')

del df_files


# calculate statistics
df = df.drop('file', axis=1)
df_mean = df.groupby(['anatomy', 'metric']).mean().reset_index()
df_std = df.groupby(['anatomy', 'metric']).std().reset_index()
df_result = df_mean.merge(df_std, on=['anatomy', 'metric'], suffixes=('_mean', '_std'))

# add average
df_avg = df_result.drop('anatomy', axis=1).groupby('metric').mean().reset_index()
df_avg['anatomy'] = 'average'
df_result = pd.concat([df_result, df_avg], ignore_index=True)

df_result['Method'] = uv_method

# save to csv
df_result.to_csv(f'evaluation/csv_files/jsrt/uv_{uv_method}.csv', index=False)

# make multi-index
df_result = df_result.set_index(['anatomy', 'metric'])

print(df_result)

import pandas as pd
import torch
from clearml import InputModel
from kornia.geometry import normalize_pixel_coordinates
from monai.metrics import DiceMetric
from torch.nn import functional as F
from tqdm import tqdm

from dataset.grazer_dataset import GrazPedWriDataset
from models.uv_unet import UVUNet
from utils import convert_list_of_uv_to_coordinates
from clearml_ids import grazer_model_ids

device = 'cuda:4' if torch.cuda.is_available() else 'cpu'

model_name = 'uv'
print(f'Evaluating {model_name}')

ds = GrazPedWriDataset('test')
cl_model = InputModel(grazer_model_ids[model_name])
model = UVUNet.load(cl_model.get_weights(), 'cpu').eval().to(device)

dsc_metric = DiceMetric(include_background=True, reduction='none', num_classes=ds.N_CLASSES)
df = pd.DataFrame(columns=['file', 'anatomy', 'metric', 'value'])

uv_values = list(ds.get_anatomical_structure_uv_values().values())
uv_values = [v.to(device, non_blocking=True) for v in uv_values]
with torch.inference_mode():
    for i, (img, lm, dist_map, seg_mask, uv_map) in enumerate(tqdm(ds, desc='Evaluating', unit='img')):
        img = img.to(device, non_blocking=True)
        lm = lm.to(device, non_blocking=True)
        dist_map = dist_map.to(device, non_blocking=True)
        seg_mask = seg_mask.to(device, non_blocking=True)
        uv_map = uv_map.to(device, non_blocking=True)
        file_stem = ds.available_file_names[i]

        seg_hat, uv_hat = model.predict(img.unsqueeze(0), mask_uv=True)
        # seg_hat, uv_hat = seg_mask.unsqueeze(0), uv_map.unsqueeze(0)

        # DICE
        dsc_value = dsc_metric(seg_hat, seg_mask.unsqueeze(0)).squeeze(0)

        # TRE
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
                {'file': file_stem, 'anatomy': anatomy, 'metric': 'dice', 'value': dsc_value[anat_idx].item() * 100}
                , index=[0])], ignore_index=True)

            df = pd.concat([df, pd.DataFrame(
                {'file': file_stem, 'anatomy': anatomy, 'metric': 'tre', 'value': tre[start_idx:end_idx].mean().item()}
                , index=[0])], ignore_index=True)

            df = pd.concat([df, pd.DataFrame(
                {'file': file_stem, 'anatomy': anatomy, 'metric': 'avg_surf_dist',
                 'value': avg_surf_dist[anat_idx, start_idx:end_idx].mean().item()}
                , index=[0])], ignore_index=True)

# cluster anatomy
df['anatomy'] = df['anatomy'].replace({'Ossa metacarpalia I': 'Metacarpals',
                                       'Ossa metacarpalia II': 'Metacarpals',
                                       'Ossa metacarpalia III': 'Metacarpals',
                                       'Ossa metacarpalia IV': 'Metacarpals',
                                       'Ossa metacarpalia V': 'Metacarpals',
                                       'Os capitatum': 'Carpals',
                                       'Os hamatum': 'Carpals',
                                       'Os lunatum': 'Carpals',
                                       'Os pisiforme': 'Carpals',
                                       'Os scaphoideum': 'Carpals',
                                       'Os triquetrum': 'Carpals',
                                       'Os trapezium': 'Carpals',
                                       'Os trapezoideum': 'Carpals',
                                       'Ulna': 'Ulna/Radius',
                                       'Radius': 'Ulna/Radius',
                                       'Epiphyse Ulna': 'Ulna/Radius',
                                       'Epiphyse Radius': 'Ulna/Radius',
                                       })

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

df_result['Method'] = 'uv'

# save to csv
df_result.to_csv(f'evaluation/csv_files/grazer/uv.csv', index=False)

# make multi-index
df_result = df_result.set_index(['anatomy', 'metric'])

print(df_result)

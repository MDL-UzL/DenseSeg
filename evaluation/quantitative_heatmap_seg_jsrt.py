import pandas as pd
import torch
from clearml import InputModel
from kornia.geometry import normalize_pixel_coordinates
from monai.metrics import DiceMetric
from torch.nn import functional as F
from tqdm import tqdm

from clearml_ids import jsrt_model_ids
from dataset.jsrt_dataset import JSRTDataset
from models.kpts_unet import KeypointSegUNet
from utils import extract_kpts_from_heatmap

ds = JSRTDataset('test', False)
cl_model = InputModel(jsrt_model_ids['heatmap_seg'])
model = KeypointSegUNet.load(cl_model.get_weights(), 'cpu').eval()

dsc_metric = DiceMetric(include_background=True, reduction='none', num_classes=ds.N_CLASSES)
df = pd.DataFrame(columns=['anatomy', 'metric', 'value'])

with torch.inference_mode():
    for img, lm, dist_map, seg in tqdm(ds, desc='Evaluating', unit='img'):
        seg_hat, heatmap_hat = model(img.unsqueeze(0))
        lm_hat = extract_kpts_from_heatmap(heatmap_hat).squeeze(0)
        heatmap_hat = heatmap_hat.squeeze(0)

        # DICE
        dsc_value = dsc_metric(seg_hat.sigmoid() > 0.5, seg.unsqueeze(0)).squeeze(0)

        # TRE
        tre = torch.linalg.vector_norm(lm - lm_hat, dim=1, ord=2) * ds.PIXEL_RESOLUTION_MM

        # AVG SURF DIST
        C, H, W = dist_map.shape
        N = lm.shape[0]
        lm_hat_norm = normalize_pixel_coordinates(lm_hat, H, W)
        dist_values = F.grid_sample(dist_map.unsqueeze(0), lm_hat_norm.view(1, 1, N, 2), align_corners=True).squeeze()
        avg_surf_dist = dist_values.abs() * ds.PIXEL_RESOLUTION_MM

        tre_organ = []
        for anat_idx, (anatomy, (start_idx, end_idx)) in enumerate(ds.get_anatomical_structure_index().items()):
            df = pd.concat([df, pd.DataFrame(
                {'anatomy': anatomy, 'metric': 'tre', 'value': tre[start_idx:end_idx].mean().item()}
                , index=[0])], ignore_index=True)

            df = pd.concat([df, pd.DataFrame(
                {'anatomy': anatomy, 'metric': 'avg_surf_dist',
                 'value': avg_surf_dist[anat_idx, start_idx:end_idx].mean().item()}
                , index=[0])], ignore_index=True)

            df = pd.concat([df, pd.DataFrame(
                {'anatomy': anatomy, 'metric': 'dice', 'value': dsc_value[anat_idx].item() * 100}
                , index=[0])], ignore_index=True)

# rename anatomy
df['anatomy'] = df['anatomy'].replace({'left_lung': 'lungs', 'right_lung': 'lungs',
                                       'heart': 'heart', 'left_clavicle': 'clavicles', 'right_clavicle': 'clavicles'})

# calculate statistics
df_mean = df.groupby(['anatomy', 'metric']).mean().reset_index()
df_std = df.groupby(['anatomy', 'metric']).std().reset_index()
df_result = df_mean.merge(df_std, on=['anatomy', 'metric'], suffixes=('_mean', '_std'))

# add average
df_avg = df_result.drop('anatomy', axis=1).groupby('metric').mean().reset_index()
df_avg['anatomy'] = 'average'
df_result = pd.concat([df_result, df_avg], ignore_index=True)

# add Method column
df_result['Method'] = 'HeatRegSeg'

# save to csv
df_result.to_csv('evaluation/csv_files/jsrt/heatmap_regression_and_seg.csv', index=False)


# make multi-index
df_result = df_result.set_index(['anatomy', 'metric'])

print(df_result)

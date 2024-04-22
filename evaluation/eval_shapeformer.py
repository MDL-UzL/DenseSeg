import pandas as pd
import torch
from kornia.geometry import denormalize_pixel_coordinates, normalize_pixel_coordinates
from skimage.draw import polygon2mask
from tqdm import tqdm
from torch.nn import functional as F

from dataset.jsrt_dataset import JSRTDataset

shape_former_results = torch.load('/home/ron/Documents/point-transformer/visualization/store/7f14a1a9230e4bd1bbfcb398ee56eca3.pt')
lms_hat = shape_former_results['pred_lm']
lms_hat = lms_hat.mean(dim=1)  # average over random initializations
lms_hat = denormalize_pixel_coordinates(lms_hat, 256, 256)

ds = JSRTDataset('test', False)
df = pd.DataFrame(columns=['anatomy', 'metric', 'value'])
for i, (_, lm, dist_map, seg) in enumerate(tqdm(ds, desc='Evaluating', unit='img')):
    lm_hat = lms_hat[i]

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

        # DICE
        seg_hat = polygon2mask((H, W), lm_hat[start_idx:end_idx].flip(-1))
        seg_hat = torch.from_numpy(seg_hat)
        intersection = (seg_hat & seg[anat_idx].bool()).sum()
        set_power = seg_hat.sum() + seg[anat_idx].sum()
        dsc = 2 * intersection / (set_power + 1e-6)

        df = pd.concat([df, pd.DataFrame(
            {'anatomy': anatomy, 'metric': 'dice', 'value': dsc.item() * 100}, index=[0])], ignore_index=True)

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
df_result['Method'] = 'ShapeFormer'

# save to csv
df_result.to_csv('evaluation/csv_files/ShapeFormer.csv', index=False)


# make multi-index
df_result = df_result.set_index(['anatomy', 'metric'])

print(df_result)

import pandas as pd
import torch
from clearml import InputModel
from kornia.geometry import normalize_pixel_coordinates
from skimage.draw import polygon2mask
from torch.nn import functional as F
from tqdm import tqdm

from clearml_ids import grazer_model_ids
from dataset.grazer_dataset import GrazPedWriDataset
from models.kpts_unet import KeypointUNet
from utils import extract_kpts_from_heatmap

model_name = 'heatmap'
print(f'Evaluating {model_name}')

ds = GrazPedWriDataset('test')
cl_model = InputModel(grazer_model_ids[model_name])
model = KeypointUNet.load(cl_model.get_weights(), 'cpu').eval()

df = pd.DataFrame(columns=['anatomy', 'metric', 'value'])


with torch.inference_mode():
    for img, lm, dist_map, seg, _ in tqdm(ds, desc='Evaluating', unit='img'):
        heatmap_hat = model(img.unsqueeze(0))
        lm_hat = extract_kpts_from_heatmap(heatmap_hat).squeeze(0)
        heatmap_hat = heatmap_hat.squeeze(0)

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

            # from matplotlib import pyplot as plt
            #
            # plt.imshow(img.squeeze().numpy(), cmap='gray')
            # plt.imshow(seg_hat, alpha=seg_hat.float())
            # plt.scatter(lm[start_idx:end_idx, 0], lm[start_idx:end_idx, 1], c='r')
            # plt.show()

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
                                       'Os trapezoideum': 'Carpals', })

# calculate statistics
df_mean = df.groupby(['anatomy', 'metric']).mean().reset_index()
df_std = df.groupby(['anatomy', 'metric']).std().reset_index()
df_result = df_mean.merge(df_std, on=['anatomy', 'metric'], suffixes=('_mean', '_std'))

# add average
df_avg = df_result.drop('anatomy', axis=1).groupby('metric').mean().reset_index()
df_avg['anatomy'] = 'average'
df_result = pd.concat([df_result, df_avg], ignore_index=True)

# add Method column
df_result['Method'] = 'Heatmap Regression'

# save to csv
# df_result.to_csv('evaluation/csv_files/grazer/heatmap_regression.csv', index=False)

# make multi-index
df_result = df_result.set_index(['anatomy', 'metric'])

print(df_result)



import pandas as pd
import torch
from clearml import InputModel
from tqdm import trange

from clearml_ids import jsrt_model_ids
from dataset.jsrt_dataset import JSRTDatasetUV
from models.uv_unet import UVUNet
from utils import convert_uv_to_coordinates

ds = JSRTDatasetUV('test', 'cartesian')
dict_lms_uv_data = torch.load('dataset/data/ablation_study_unknown_lms_uv.pth')
# apply split
for anatomy, data in dict_lms_uv_data.items():
    for key, value in data.items():
        data[key]['lms'] = value['lms'][ds.SPLIT_IDX:]

cl_model = InputModel(jsrt_model_ids['uv'])
model = UVUNet.load(cl_model.get_weights(), 'cpu').eval()

df = pd.DataFrame(columns=['file', 'anatomy', 'tre'])

with torch.inference_mode():
    for i, (anatomy, lms_uv_dict) in enumerate(dict_lms_uv_data.items()):
        lms = torch.cat([lms_uv_dict['clavicles']['lms'], lms_uv_dict['center']['lms']], dim=1)
        for ds_idx in trange(len(ds)):
            img = ds[ds_idx][0]
            _, uv_hat = model.predict(img.unsqueeze(0), mask_uv=True)

            # TRE
            uv_values = torch.cat([lms_uv_dict['clavicles']['uv_values'], lms_uv_dict['center']['uv_values']], 0)
            lm_hat = convert_uv_to_coordinates(uv_hat[:, i], uv_values.unsqueeze(0), 'linear', k=5)
            tre = torch.linalg.vector_norm(lms[i] - lm_hat, dim=-1, ord=2).squeeze(0)
            tre *= ds.PIXEL_RESOLUTION_MM

            df = pd.concat(
                [df,
                 pd.DataFrame({'file': i, 'anatomy': anatomy + '_4ClvLms', 'tre': tre[:-1].mean().item()}, index=[0])],
                ignore_index=True)
            df = pd.concat(
                [df, pd.DataFrame({'file': i, 'anatomy': anatomy + '_center', 'tre': tre[-1].item()}, index=[0])],
                ignore_index=True)

# cluster anatomy
df['anatomy'] = df['anatomy'].replace({'right_lung_4ClvLms': 'clav', 'left_lung_4ClvLms': 'clav',
                                       'right_lung_center': 'center', 'left_lung_center': 'center'})

# calculate statistics
df = df.drop('file', axis=1)
df_mean = df.groupby('anatomy').mean().reset_index()
df_std = df.groupby('anatomy').std().reset_index()
df_result = df_mean.merge(df_std, on=['anatomy'], suffixes=('_mean', '_std'))

df_result = df_result.set_index(['anatomy'])

print(df_result)

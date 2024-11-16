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
for data in dict_lms_uv_data.values():
    data['lms'] = data['lms'][ds.SPLIT_IDX:]

cl_model = InputModel(jsrt_model_ids['uv'])
model = UVUNet.load(cl_model.get_weights(), 'cpu').eval()

df = pd.DataFrame(columns=['file', 'study', 'tre'])

with torch.inference_mode():
    for study_name, lms_uv_dict in enumerate(dict_lms_uv_data.items()):
        lms = lms_uv_dict['lms']
        for ds_idx in trange(len(ds)):
            img = ds[ds_idx][0]
            _, uv_hat = model.predict(img.unsqueeze(0), mask_uv=True)

            # TRE
            uv_values = lms_uv_dict['uv_values']
            lm_hat = convert_uv_to_coordinates(uv_hat[:, 1], uv_values.unsqueeze(0), 'linear', k=5)
            tre = torch.linalg.vector_norm(lms[ds_idx] - lm_hat, dim=-1, ord=2).squeeze(0)
            tre *= ds.PIXEL_RESOLUTION_MM

            df = pd.concat(
                [df,
                 pd.DataFrame({'file': ds_idx, 'study': study_name, 'tre': tre.mean().item()}, index=[0])],
                ignore_index=True)

# calculate statistics
df = df.drop('file', axis=1)
df_mean = df.groupby('study').mean().reset_index()
df_std = df.groupby('study').std().reset_index()
df_result = df_mean.merge(df_std, on=['study'], suffixes=('_mean', '_std'))

df_result = df_result.set_index(['study'])

print(df_result)

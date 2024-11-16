import torch
from clearml import InputModel
from matplotlib import pyplot as plt
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

            # Plot
            plt.figure(figsize=(10, 10))
            plt.imshow(img.squeeze(), cmap='gray')
            plt.scatter(lms[i, :, 0], lms[i, :, 1], c='r', label='Ground Truth')
            plt.scatter(lm_hat[0, :, 0], lm_hat[0, :, 1], c='b', label='Prediction')
            plt.title(f'{ds_idx} {anatomy} - TRE: {tre.mean().item():.4f} mm')
            plt.legend()

            if ds_idx >= 3:
                break

plt.show()

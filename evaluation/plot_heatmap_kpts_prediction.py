from random import randint

import torch
from clearml import InputModel
from matplotlib import pyplot as plt

from dataset.jsrt_dataset import JSRTDataset
from models.kpts_unet import KeypointUNet
from utils import extract_kpts_from_heatmap
from clearml_ids import jsrt_model_ids

ds = JSRTDataset('test', False)
cl_model = InputModel(jsrt_model_ids['heatmap'])
model = KeypointUNet.load(cl_model.get_weights(), 'cpu').eval()

rnd_idx = randint(0, len(ds) - 1)
img, lm, _, _ = ds[rnd_idx]

plt.imshow(img.squeeze(), cmap='gray')

with torch.inference_mode():
    heatmap_hat = model(img.unsqueeze(0))
    lm_hat = extract_kpts_from_heatmap(heatmap_hat).squeeze(0)
    heatmap_hat = heatmap_hat.squeeze(0)

for a_idx, (anatomy, (start_idx, end_idx)) in enumerate(ds.get_anatomical_structure_index().items()):
    plt.figure(anatomy)

    # extract bounding box
    min_idx = lm[start_idx:end_idx].min(dim=1).values * 0.85
    max_idx = lm[start_idx:end_idx].max(dim=1).values * 1.15

    plt.imshow(img.squeeze(), cmap='gray')
    plt.scatter(lm[start_idx:end_idx, 0], lm[start_idx:end_idx, 1], c='g', label='Ground Truth', marker='x')
    plt.scatter(lm_hat[start_idx:end_idx, 0], lm_hat[start_idx:end_idx, 1], c='b', label='Prediction')
    plt.legend()
    # plt.xlim(min_idx[0].item(), max_idx[1].item())
    # plt.ylim(min_idx[0].item(), max_idx[1].item())
    # plt.gca().invert_yaxis()

    # plt.figure(f'{anatomy} Heatmap')
    # plt.imshow(heatmap_hat[a_idx], cmap='jet')
    # plt.colorbar()

plt.show()

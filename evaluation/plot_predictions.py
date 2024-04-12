from pathlib import Path
from random import randint

import torch
from clearml import InputModel
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from dataset.jsrt_dataset import JSRTDatasetUV
from models.uv_unet import UVUNet
from utils import convert_uv_to_coordinates

save_plt = False
save_path = Path('/home/ron/Desktop')

ds = JSRTDatasetUV('test')
cl_model = InputModel('89fe0a03364f4ab492b9508a8de47623')
model = UVUNet.load(cl_model.get_weights(), 'cpu').eval()

rnd_idx = randint(0, len(ds) - 1)
img, shape, _, seg_mask, uv_map = ds[rnd_idx]

plt.imshow(img.squeeze(), cmap='gray')
if save_plt:
    plt.savefig(save_path / 'input.png')

with torch.inference_mode():
    seg_hat, uv_hat = model.predict(img.unsqueeze(0), mask_uv=True)
    seg_hat = seg_hat.squeeze().float()
    uv_hat = uv_hat.squeeze()

for a_idx, (anatomy, (start_idx, end_idx)) in enumerate(ds.get_anatomical_structure_index().items()):
    fig, axs = plt.subplots(3, 3)
    fig.suptitle(anatomy)

    # extract bounding box
    mask = seg_mask[a_idx].nonzero()
    min_idx = mask.min(dim=0).values * 0.85
    max_idx = mask.max(dim=0).values * 1.15

    for a in axs.flatten():
        a.imshow(img.squeeze(), cmap='gray')
        a.set_xlim(min_idx[1], max_idx[1])
        a.set_ylim(min_idx[0], max_idx[0])
        a.invert_yaxis()

    axs[0, 0].imshow(seg_mask[a_idx], alpha=seg_mask[a_idx], interpolation='nearest')
    axs[0, 0].set_title('Ground Truth')
    axs[0, 1].imshow(seg_hat[a_idx], alpha=seg_hat[a_idx], interpolation='nearest')
    axs[0, 1].set_title('Prediction')
    diff0 = axs[0, 2].imshow(seg_mask[a_idx] - seg_hat[a_idx], alpha=abs(seg_mask[a_idx] - seg_hat[a_idx]))
    axs[0, 2].set_title('Difference')
    # colorbar
    divider = make_axes_locatable(axs[0, 2])
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(diff0, cax=cax, orientation='vertical')

    axs[1, 0].imshow(uv_map[a_idx, 0], cmap='viridis')
    axs[1, 1].imshow(uv_hat[a_idx, 0], cmap='viridis')
    diff1 = axs[1, 2].imshow(uv_map[a_idx, 0] - uv_hat[a_idx, 0], cmap='coolwarm')
    # colorbar
    divider = make_axes_locatable(axs[1, 2])
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(diff1, cax=cax, orientation='vertical')

    axs[2, 0].imshow(uv_map[a_idx, 1], cmap='viridis')
    axs[2, 1].imshow(uv_hat[a_idx, 1], cmap='viridis')
    diff2 = axs[2, 2].imshow(uv_map[a_idx, 1] - uv_hat[a_idx, 1], cmap='coolwarm')
    # colorbar
    divider = make_axes_locatable(axs[2, 2])
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(diff2, cax=cax, orientation='vertical')

    fig.tight_layout()
    if save_plt:
        plt.savefig(save_path / f'{anatomy}_seg_uv.png')

    gt_lms = shape[start_idx:end_idx]
    # convert uv to coordinates
    cnvt_func = lambda uv_map, uv_val: convert_uv_to_coordinates(uv_map.unsqueeze(0), uv_val.unsqueeze(0), 'linear', 5).squeeze(0)
    lms = cnvt_func(uv_map[a_idx], ds.get_anatomical_structure_uv_values()[anatomy])
    lms_hat = cnvt_func(uv_hat[a_idx], ds.get_anatomical_structure_uv_values()[anatomy])
    color_code = torch.arange(len(gt_lms))

    fig, axs = plt.subplots(1, 3)
    fig.suptitle(anatomy)
    for a in axs.flatten():
        a.imshow(img.squeeze(), cmap='gray')
        a.set_xlim(min_idx[1], max_idx[1])
        a.set_ylim(min_idx[0], max_idx[0])
        a.invert_yaxis()

    axs[0].scatter(gt_lms[:, 0], gt_lms[:, 1], c=color_code, cmap='tab20')
    axs[0].set_title('Ground Truth')
    axs[1].scatter(lms[:, 0], lms[:, 1], c=color_code, cmap='tab20')
    axs[1].set_title('Coords from GT UV')
    axs[2].scatter(lms_hat[:, 0], lms_hat[:, 1], c=color_code, cmap='tab20')
    axs[2].set_title('Coords from predicted UV')

    fig.tight_layout()
    if save_plt:
        plt.savefig(save_path / f'{anatomy}_coords.png')

plt.show()

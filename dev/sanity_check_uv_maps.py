from dataset.jsrt_dataset import JSRTDatasetUV
from matplotlib import pyplot as plt
from random import randint
from utils import convert_uv_to_coordinates

ds_cartesian = JSRTDatasetUV('train', uv_mode='cartesian')
ds_polar = JSRTDatasetUV('train', uv_mode='polar')

idx = randint(0, len(ds_cartesian) - 1)
img, target_lm, _, _, uv_map_cartesian = ds_cartesian[idx]
_, _, _, _, uv_map_polar = ds_polar[idx]

for a_idx, (anatomy, (start_idx, end_idx)) in enumerate(ds_cartesian.get_anatomical_structure_index().items()):
    lm_cartesian = convert_uv_to_coordinates(uv_map_cartesian[a_idx].unsqueeze(0),
                                             ds_cartesian.get_anatomical_structure_uv_values()[anatomy].unsqueeze(0),
                                             'linear', 5).squeeze()
    lm_polar = convert_uv_to_coordinates(uv_map_polar[a_idx].unsqueeze(0),
                                         ds_polar.get_anatomical_structure_uv_values()[anatomy].unsqueeze(0), 'linear',
                                         5).squeeze()

    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    ax[0].imshow(img.squeeze(), cmap='gray')
    ax[0].scatter(target_lm[start_idx:end_idx, 0], target_lm[start_idx:end_idx, 1], c='r')
    ax[0].set_title('Image')

    ax[1].imshow(uv_map_cartesian[a_idx, 0])
    ax[1].scatter(target_lm[start_idx:end_idx, 0], target_lm[start_idx:end_idx, 1], c='r')
    ax[1].set_title('U map (cartesian)')

    ax[2].imshow(uv_map_polar[a_idx, 0])
    ax[2].scatter(target_lm[start_idx:end_idx, 0], target_lm[start_idx:end_idx, 1], c='r')
    ax[2].set_title('U map (polar)')

    fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    ax[0].scatter(target_lm[start_idx:end_idx, 0], target_lm[start_idx:end_idx, 1])
    ax[0].scatter(lm_cartesian[:, 0], lm_cartesian[:, 1])
    ax[0].set_title('Cartesian')
    ax[0].invert_yaxis()

    ax[1].scatter(target_lm[start_idx:end_idx, 0], target_lm[start_idx:end_idx, 1])
    ax[1].scatter(lm_polar[:, 0], lm_polar[:, 1])
    ax[1].set_title('Polar')
    ax[1].invert_yaxis()

plt.show()

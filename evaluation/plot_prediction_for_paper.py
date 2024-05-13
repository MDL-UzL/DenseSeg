import torch
from clearml import InputModel
from kornia.geometry import normalize_pixel_coordinates, denormalize_pixel_coordinates
from matplotlib import pyplot as plt
from monai.metrics import DiceMetric
from skimage.draw import polygon2mask
from torch.nn import functional as F

from clearml_ids import grazer_model_ids, jsrt_model_ids
from dataset.grazer_dataset import GrazPedWriDataset
from dataset.jsrt_dataset import JSRTDatasetUV
from models.kpts_unet import KeypointUNet
from models.uv_unet import UVUNet
from utils import convert_list_of_uv_to_coordinates, sort_kpts_clockwise, extract_kpts_from_heatmap

example = ['best', 'median', 'worst'][2]
model_name = ['uv', 'heatmap', 'shapeformer'][2]
ds_2_use = ['jsrt', 'grazer'][0]
print(f'Example: {example}, Model: {model_name}, Dataset: {ds_2_use}')

if ds_2_use == 'grazer':
    ds = GrazPedWriDataset('test')
    idx_to_use = {
        'best': ds.available_file_names.index('0417_0697542554_01_WRI-R1_F008'),
        'median': ds.available_file_names.index('1219_0712082244_02_WRI-L1_F011'),
        'worst': ds.available_file_names.index('5082_0267021551_05_WRI-L1_F008')
    }[example]
elif ds_2_use == 'jsrt':
    ds = JSRTDatasetUV('test', 'cartesian')
    idx_to_use = {'best': 12, 'median': 82, 'worst': 0}[example]

cl_model = InputModel(grazer_model_ids[model_name] if ds_2_use == 'grazer' else jsrt_model_ids[model_name])
if model_name == 'uv':
    model = UVUNet.load(cl_model.get_weights(), 'cpu').eval()
elif model_name == 'heatmap':
    model = KeypointUNet.load(cl_model.get_weights(), 'cpu').eval()
elif model_name == 'shapeformer':
    shape_former_results = torch.load(
        '/home/ron/Documents/point-transformer/visualization/store/7f14a1a9230e4bd1bbfcb398ee56eca3.pt')
    lms_hat = shape_former_results['pred_lm']
    lms_hat = lms_hat.mean(dim=1)  # average over random initializations
    lms_hat = denormalize_pixel_coordinates(lms_hat, 256, 256)
else:
    raise ValueError('Invalid model name')

dsc_metric = DiceMetric(include_background=True, reduction='none', num_classes=ds.N_CLASSES)

with torch.inference_mode():
    img, lm, dist_map, seg_mask, uv_map = ds[idx_to_use]
    img = img.unsqueeze(0)
    if model_name == 'uv':
        seg_hat, uv_hat = model.predict(img, mask_uv=True)
        lm_hat = convert_list_of_uv_to_coordinates(
            uv_hat, list(ds.get_anatomical_structure_uv_values().values()), 'linear', k=5)
        lm_hat = torch.cat(lm_hat, dim=1).squeeze(0)
    elif model_name in ['heatmap', 'shapeformer']:
        if model_name == 'heatmap':
            heatmap_hat = model(img)
            lm_hat = extract_kpts_from_heatmap(heatmap_hat).squeeze(0)
        elif model_name == 'shapeformer':
            lm_hat = lms_hat[idx_to_use]

        seg_hat = torch.empty_like(seg_mask).unsqueeze(0)
        for anat_idx, (anatomy, (start_idx, end_idx)) in enumerate(ds.get_anatomical_structure_index().items()):
            curr_lm = sort_kpts_clockwise(lm_hat[start_idx:end_idx].float())

            H, W = img_shape = img.shape[-2:]
            mask = polygon2mask((H, W), curr_lm.flip(-1))
            seg_hat[:, anat_idx] = torch.from_numpy(mask)

    dsc_value = dsc_metric(seg_hat, seg_mask.unsqueeze(0)).squeeze(0) * 100
    tre = torch.linalg.vector_norm(lm - lm_hat, dim=1, ord=2)
    tre *= ds.PIXEL_RESOLUTION_MM

    # AVG SURF DIST
    C, H, W = dist_map.shape
    N = lm.shape[0]
    lm_hat_norm = normalize_pixel_coordinates(lm_hat, H, W)
    dist_values = F.grid_sample(dist_map.unsqueeze(0), lm_hat_norm.view(1, 1, N, 2), align_corners=True).squeeze()
    avg_surf_dist = dist_values.abs() * ds.PIXEL_RESOLUTION_MM

plt.figure(figsize=(10, 10))
plt.imshow(img.squeeze(), cmap='gray')
cmap = plt.get_cmap('tab20b')
tre_avg = torch.full((C,), torch.nan)
asd_avg = torch.full((C,), torch.nan)
for anat_idx, (anatomy, (start_idx, end_idx)) in enumerate(ds.get_anatomical_structure_index().items()):
    lm_i = lm[start_idx:end_idx]
    lm_hat_i = lm_hat[start_idx:end_idx]

    plt.scatter(lm_i[:, 0], lm_i[:, 1], c='r', marker='.')
    if ds_2_use == 'grazer':
        plt.scatter(lm_hat_i[:, 0], lm_hat_i[:, 1], color=cmap(anat_idx), marker='o')
    else:
        plt.scatter(lm_hat_i[:, 0], lm_hat_i[:, 1], marker='o')

    tre_avg[anat_idx] = tre[start_idx:end_idx].mean()
    asd_avg[anat_idx] = avg_surf_dist[anat_idx, start_idx:end_idx].mean()

print('DSC:', round(dsc_value.mean().item(), 1), '±', round(dsc_value.std().item(), 1))
print('TRE:', round(tre_avg.mean().item(), 1), '±', round(tre_avg.std().item(), 1))
print('ASD:', round(asd_avg.mean().item(), 1), '±', round(asd_avg.std().item(), 1))

plt.axis('off')
plt.savefig(f'/home/ron/Documents/Konferenzen/IJCARS/results/{ds_2_use}_{model_name}_{example}.png',
            dpi=300, bbox_inches='tight', pad_inches=0)
plt.show()

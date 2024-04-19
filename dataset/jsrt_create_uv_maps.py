import torch

from dataset.jsrt_dataset import JSRTDataset
from utils import extract_warped_uv_maps, extract_polar_uv_maps

generation_mode = 'cartesian'
print(f'Generating {generation_mode} uv maps...')
extract_fn = {'cartesian': extract_warped_uv_maps, 'polar': extract_polar_uv_maps}[generation_mode]

ds = JSRTDataset('all', normalize_landmarks=False)
mean_shape = JSRTDataset('train', normalize_landmarks=False).lms.mean(dim=0)
shapes = ds.lms
segmentations = ds.seg_masks
anatomy_idx = JSRTDataset.get_anatomical_structure_index()

N, A, H, W = segmentations.shape

uv_maps = torch.empty((N, A, 2, H, W), dtype=torch.float32)
mean_shape_uv_values = dict()

for a, (anatomy, (start_idx, end_idx)) in enumerate(anatomy_idx.items()):
    uv_maps[:, a], uv_values = extract_fn(mean_shape[start_idx:end_idx], shapes[:, start_idx:end_idx], segmentations[:, a])
    mean_shape_uv_values[anatomy] = uv_values

torch.save(uv_maps, f'dataset/data/uv_maps_{generation_mode}.pth')
torch.save(mean_shape_uv_values, f'dataset/data/mean_shape_uv_values_{generation_mode}.pth')

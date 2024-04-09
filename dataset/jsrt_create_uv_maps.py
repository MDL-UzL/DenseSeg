import torch

from dataset import JSRTDataset
from utils import extract_warped_uv_maps

ds = JSRTDataset('all', normalize_landmarks=False)
mean_shape = JSRTDataset('train', normalize_landmarks=False).lms.mean(dim=0)
shapes = ds.lms
segmentations = ds.seg_masks
anatomy_idx = JSRTDataset.get_anatomical_structure_index()

N, A, H, W = segmentations.shape

uv_maps = torch.empty((N, A, 2, H, W), dtype=torch.float32)
mean_shape_uv_values = dict()

for a, (anatomy, (start_idx, end_idx)) in enumerate(anatomy_idx.items()):
    uv_maps[:, a], uv_values = extract_warped_uv_maps(mean_shape[start_idx:end_idx], shapes[:, start_idx:end_idx],
                                           segmentations[:, a])
    mean_shape_uv_values[anatomy] = uv_values

torch.save(uv_maps, 'dataset/data/uv_maps.pth')
torch.save(mean_shape_uv_values, 'dataset/data/mean_shape_uv_values.pth')

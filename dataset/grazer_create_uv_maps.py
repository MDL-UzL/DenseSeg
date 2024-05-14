from collections import OrderedDict

import h5py
import pandas as pd
import torch

from utils import extract_warped_uv_maps, extract_polar_uv_maps

generation_mode = 'polar'
print(f'Generating {generation_mode} uv maps...')
assert generation_mode == 'polar', 'Only polar mode is supported for Graz dataset'
extract_fn = {'cartesian': extract_warped_uv_maps, 'polar': extract_polar_uv_maps}[generation_mode]

storage = h5py.File('/home/ron/Documents/DenseSeg/dataset/data/graz/graz_img_seg_lms.h5', 'r')
csv_path = '/home/ron/Documents/DenseSeg/dataset/data/graz/dataset_with_cv_split.csv'
df_meta = pd.read_csv(csv_path)
df_meta = df_meta[df_meta['filestem'].isin(storage.keys())]
df_train = df_meta[(df_meta['cv_test_idx'] == 0) | (df_meta['cv_test_idx'] == 2)]

# calculate mean shape over training split
shapes = [torch.tensor(storage[f]['lms'][:]) for f in df_train['filestem']]
shapes = torch.stack(shapes)
mean_shape = shapes.mean(dim=0)

shapes = [torch.tensor(storage[f]['lms'][:]) for f in df_meta['filestem']]
shapes = torch.stack(shapes)

segmentations = torch.stack([torch.tensor(storage[f]['seg'][:]) for f in df_meta['filestem']])
anatomy_idx = OrderedDict()
idx = 0
for organ, num_lms in zip(storage.attrs['BONE_LABEL'], storage.attrs['NUM_LMS']):
    anatomy_idx[organ] = (idx, idx + num_lms)
    idx += num_lms

N, A, H, W = segmentations.shape
# clip shapes to image size
shapes[..., 0] = shapes[..., 0].clip(0, W - 1)
shapes[..., 1] = shapes[..., 1].clip(0, H - 1)

uv_maps = torch.empty((N, A, 2, H, W), dtype=torch.float32)
mean_shape_uv_values = dict()

for a, (anatomy, (start_idx, end_idx)) in enumerate(anatomy_idx.items()):
    uv_maps[:, a], uv_values = extract_fn(mean_shape[start_idx:end_idx], shapes[:, start_idx:end_idx],
                                          segmentations[:, a])
    mean_shape_uv_values[anatomy] = uv_values

uv_maps = {f: uv_map for f, uv_map in zip(df_meta['filestem'], uv_maps)}
torch.save(uv_maps, f'dataset/data/graz/uv_maps_{generation_mode}.pth')
torch.save(mean_shape_uv_values, f'dataset/data/graz/mean_shape_uv_values_{generation_mode}.pth')

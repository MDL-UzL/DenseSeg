import logging

import torch
from utils import farthest_point_sampling
from dataset.grazer_dataset import GrazPedWriDataset
from tqdm import tqdm

ds = GrazPedWriDataset('train')
mean_shape = ds.mean_shape
num_lms = ds.num_lms
total_num_lms = sum(num_lms)

mask_num_lms_dict = {}
for percentage in tqdm([.25, .5, .75]):
    num_kpts = torch.from_numpy(num_lms) * percentage
    num_kpts = num_kpts.round().int()

    if any(num_kpts < 4):
        num_kpts = num_kpts.clamp_min(4)
        logging.warning('Minimum number of landmark is reached for some bones. Set them to 4.')

    selected_lms = torch.zeros(total_num_lms, dtype=bool)
    for (start_idx, end_idx), n in zip(ds.get_anatomical_structure_index().values(), num_kpts):
        _, ind = farthest_point_sampling(mean_shape[start_idx:end_idx].unsqueeze(0), n)
        selected_lms[start_idx:end_idx][ind] = True

    mask_num_lms_dict[percentage] = {
        'mask': selected_lms,
        'num_lms': num_kpts.numpy()
    }

torch.save(mask_num_lms_dict, 'dataset/data/graz/graz_lms_mask_dict.pth')

from pathlib import Path
import h5py
import torch
from tqdm import tqdm
import cv2
from scipy.ndimage import distance_transform_edt
import numpy as np

seg_path = Path('/home/ron/Documents/DenseSeg/dataset/data/graz/raw_segmentations_no_cast.h5')
lms_path = Path('/home/ron/Documents/DenseSeg/dataset/data/graz/lms_dsc_900.pth')
img_path = Path('/home/ron/Documents/SemiSAM/data/img_only_front_all_left')

lms_dict = torch.load(lms_path, map_location='cpu')
available_files = set(lms_dict['keys'])
lms_storage = lms_dict['lms']
seg_storage = h5py.File(seg_path, 'r')
H, W = 384, 224
IMG_MEAN = 0.3505533917353781
IMG_STD = 0.22763733675869177
NUM_LMS = [40, 20, 35, 25, 30, 30, 30, 30, 25, 30, 50, 50, 50, 50, 60, 80, 75]
BONE_LABEL = sorted([
    'Radius',
    'Ulna',
    'Os scaphoideum',
    'Os lunatum',
    'Os triquetrum',
    'Os pisiforme',
    'Os trapezium',
    'Os trapezoideum',
    'Os capitatum',
    'Os hamatum',
    'Ossa metacarpalia I',
    'Ossa metacarpalia II',
    'Ossa metacarpalia III',
    'Ossa metacarpalia IV',
    'Ossa metacarpalia V',
    'Epiphyse Radius',
    'Epiphyse Ulna'])
assert len(BONE_LABEL) == len(NUM_LMS), 'Number of labels and number of landmarks do not match'

graz_img_seg_lms_storage = h5py.File('/home/ron/Documents/DenseSeg/dataset/data/graz/graz_img_seg_lms.h5', 'w')
graz_img_seg_lms_storage.attrs['IMG_MEAN'] = IMG_MEAN
graz_img_seg_lms_storage.attrs['IMG_STD'] = IMG_STD
graz_img_seg_lms_storage.attrs['NUM_LMS'] = torch.tensor(NUM_LMS).numpy()
graz_img_seg_lms_storage.attrs['BONE_LABEL'] = BONE_LABEL

for file in tqdm(available_files):
    img = cv2.imread(str(img_path / (file + '.png')), cv2.IMREAD_GRAYSCALE).astype(float) / 255
    img = cv2.resize(img, (W, H), interpolation=cv2.INTER_LINEAR)
    img = (img - IMG_MEAN) / IMG_STD

    seg = seg_storage['segmentation_mask'][file]
    assert seg.shape == (len(NUM_LMS), H, W), f'Shape of segmentation does not match for {file}'
    # calculate signed distance map
    dist_map = np.empty((len(NUM_LMS), H, W), dtype=np.float32)
    for i, s in enumerate(seg):
        dist_map_a = torch.tensor(distance_transform_edt(s))
        dist_map_b = torch.tensor(distance_transform_edt(np.logical_not(s)))
        dist_map[i] = dist_map_b - dist_map_a
    assert seg.shape == dist_map.shape, f'Shape of segmentation and distance map does not match for {file}'

    lms_storage_idx = lms_dict['keys'].index(file)
    lms = lms_storage[lms_storage_idx]
    assert [len(l) for l in lms] == NUM_LMS, f'Number of landmarks do not match for {file}'
    # correct symmetrical padding for registration
    lms = torch.cat(lms, dim=0).flip(-1).numpy() - 32

    storage = graz_img_seg_lms_storage.create_group(file)
    storage.create_dataset('img', data=img)
    storage.create_dataset('seg', data=seg)
    storage.create_dataset('dist_map', data=dist_map)
    storage.create_dataset('lms', data=lms)

graz_img_seg_lms_storage.close()

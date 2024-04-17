from torch.utils.data import Dataset
import h5py
import json
from pathlib import Path
from torch.nn import functional as F
import torch
import cv2
import pandas as pd
import random


class GrazPedWriDataset(Dataset):
    ## calculated over training split
    IMG_MEAN = 0.3505533917353781
    IMG_STD = 0.22763733675869177
    # bone label
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
    BONE_LABEL_MAPPING = {k: v for k, v in zip(BONE_LABEL, range(len(BONE_LABEL)))}
    N_CLASSES = len(BONE_LABEL)

    def __init__(self, mode:str, saved_seg_path: str, img_path: str, csv_path: str, rescale_HW: tuple = (384, 224)):
        """
        Dataset that loads images with their stored segmentation. Not include images from training or testing split.
        :param saved_seg_path: path to saved segmentation. Has to be h5 file.
        :param img_path: path to images
        :param csv_path: path to csv file with meta information. datsaset.csv
        :param rescale_HW: rescale image and ground truth. None will not rescale.
        """
        super().__init__()

        # load data meta and other information
        self.df_meta = pd.read_csv(csv_path, index_col='filestem')
        h5_file = h5py.File(saved_seg_path, 'r')
        lbl_loaded = json.loads(h5_file.attrs['labels'])
        assert lbl_loaded == self.BONE_LABEL_MAPPING, 'Loaded labels do not match'

        # load data meta and other information
        self.img_path = Path(img_path)
        self.ds_saved_seg = h5_file['segmentation_mask']

        # get file names
        random.seed(42)
        self.available_file_names = list(self.ds_saved_seg.keys())
        random.shuffle(self.available_file_names)
        if mode == 'train':
            self.available_file_names = self.available_file_names[:int(0.8 * len(self.available_file_names))]
        elif mode == 'test':
            self.available_file_names = self.available_file_names[int(0.8 * len(self.available_file_names)):]
        else:
            raise ValueError(f"Unknown mode {mode}. Has to be 'train' or 'test'")

        # init transformation
        self.resize_lbl = lambda x: F.interpolate(x.float().unsqueeze(0), size=rescale_HW, mode='nearest').squeeze(0)

    def __len__(self):
        return len(self.available_file_names)

    def __getitem__(self, index) -> (torch.Tensor, torch.Tensor, str):
        """
        get item by index
        :param index: index of item
        :return: image, segmentation, age in months, gender (M=1, F=0), file name
        """
        file_name = self.available_file_names[index]
        age_in_months = self.df_meta.loc[file_name, 'age'] * 12
        gender = self.df_meta.loc[file_name, 'gender']
        gender = 1 if gender == 'M' else 0  # M:1, F:0

        # segmentation mask
        seg_masks = torch.from_numpy(self.ds_saved_seg[file_name][:])
        seg_masks = self.resize_lbl(seg_masks)

        # numpy image to tensor and add channel dimension
        img = cv2.imread(str(self.img_path.joinpath(file_name).with_suffix('.png')), cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, seg_masks.shape[-2:][::-1], interpolation=cv2.INTER_LINEAR)
        img = torch.from_numpy(img).unsqueeze(0).float()
        img /= 255

        return img, seg_masks, age_in_months, gender, file_name

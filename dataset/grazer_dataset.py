import h5py
import pandas as pd
import torch
from torch.utils.data import Dataset


class GrazPedWriDataset(Dataset):
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
    BCE_POS_WEIGHTS = torch.tensor([108.1348, 349.1551, 69.6342, 96.0886, 167.7897, 364.5914, 131.5362,
                                    176.2591, 240.9182, 169.5408, 60.1363, 46.6512, 51.6916, 58.6216,
                                    52.5956, 11.2623, 17.9409])
    BCE_POS_WEIGHTS = BCE_POS_WEIGHTS.view(N_CLASSES, 1, 1).expand(N_CLASSES, 384, 224)
    CLASS_LABEL = BONE_LABEL

    def __init__(self, mode: str):
        super().__init__()

        # load data meta and other information
        self.storage = h5py.File('dataset/data/graz/graz_img_seg_lms.h5', 'r')
        assert self.storage.attrs['BONE_LABEL'].tolist() == self.BONE_LABEL, 'Loaded labels do not match'
        self.uv_maps = torch.load('dataset/data/graz/uv_maps_polar.pth')
        self.uv_values = torch.load('dataset/data/graz/mean_shape_uv_values_polar.pth')
        self.df_meta = pd.read_csv('dataset/data/graz/dataset_with_cv_split.csv', index_col='filestem')
        self.df_meta = self.df_meta[self.df_meta.index.isin(self.storage.keys())]

        # get file names
        if mode == 'train':
            file_mask = (self.df_meta['cv_test_idx'] == 0) | (self.df_meta['cv_test_idx'] == 2)
        elif mode == 'test':
            file_mask = (self.df_meta['cv_test_idx'] == -1) | (self.df_meta['cv_test_idx'] == 1)
        else:
            raise ValueError(f"Unknown mode {mode}. Has to be 'train' or 'test'")
        self.available_file_names = self.df_meta[file_mask].index.tolist()

    def __len__(self):
        return len(self.available_file_names)

    def __getitem__(self, index):
        file_name = self.available_file_names[index]
        ds = self.storage[file_name]

        img = torch.from_numpy(ds['img'][:]).float().unsqueeze(0)  # add channel dimension
        seg = torch.from_numpy(ds['seg'][:]).float()
        lms = torch.from_numpy(ds['lms'][:])
        uv_map = self.uv_maps[file_name]

        return img, lms, -1, seg, uv_map  # -1 placeholder for dist_map

    def get_anatomical_structure_uv_values(self):
        return self.uv_values


if __name__ == '__main__':
    from matplotlib import pyplot as plt

    ds = GrazPedWriDataset('train')
    img, lms, _, seg, uv_map = ds[0]

    plt.imshow(img.squeeze(), cmap='gray')
    plt.figure()
    plt.imshow(seg[0])
    plt.figure()
    plt.imshow(uv_map[0, 0])
    plt.scatter(lms[:, 0], lms[:, 1])
    plt.show()

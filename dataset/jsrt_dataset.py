import logging
from collections import OrderedDict

import torch
from torch.utils.data import Dataset


class JSRTDataset(Dataset):
    SPLIT_IDX = 160
    PIXEL_RESOLUTION_MM = 1.4
    NUM_LANDMARKS = {'right_lung': 44, 'left_lung': 50, 'heart': 26, 'right_clavicle': 23, 'left_clavicle': 23}
    CLASS_LABEL = list(NUM_LANDMARKS.keys())
    N_CLASSES = len(NUM_LANDMARKS)
    BCE_POS_WEIGHTS = torch.tensor([4.9105, 6.3263, 10.2737, 106.2339, 106.7618])
    BCE_POS_WEIGHTS = BCE_POS_WEIGHTS.view(N_CLASSES, 1, 1).expand(N_CLASSES, 256, 256)

    def __init__(self, mode: str, normalize_landmarks: bool = True):
        super().__init__()

        # load data
        data = torch.load('dataset/data/JSRT_img0_lms.pth', map_location='cpu')
        self.imgs = data['JSRT_img0'].float()  # already z-normalized
        self.lms = data['JSRT_lms'].float()
        self.dist_map = torch.load('dataset/data/jsrt_distmaps.pth').float()
        del data
        self.seg_masks = torch.load('dataset/data/jsrt_seg_masks.pth').float()

        # check for equal number of samples
        assert self.imgs.shape[0] == self.lms.shape[0] == self.dist_map.shape[0]

        # normalize landmarks to [-1, 1]
        assert self.imgs.shape[-1] == self.imgs.shape[-2] == 256, f'Expected image size 256, but got {self.imgs.shape}'
        assert self.lms.max() <= 256 and self.lms.min() >= 0
        if normalize_landmarks:
            self.lms = self.lms / 256 * 2 - 1

        # select images for training or testing
        if mode == 'train':
            self.imgs = self.imgs[:self.SPLIT_IDX]
            self.lms = self.lms[:self.SPLIT_IDX]
            self.dist_map = self.dist_map[:self.SPLIT_IDX]
            self.seg_masks = self.seg_masks[:self.SPLIT_IDX]
        elif mode == 'test':
            self.imgs = self.imgs[self.SPLIT_IDX:]
            self.lms = self.lms[self.SPLIT_IDX:]
            self.dist_map = self.dist_map[self.SPLIT_IDX:]
            self.seg_masks = self.seg_masks[self.SPLIT_IDX:]
        elif mode == 'all':
            logging.warning('Using all data. Please do not use this mode in context of training.')
        else:
            raise ValueError(f'Unknown mode {mode}')

    def __len__(self):
        return self.imgs.shape[0]

    def __getitem__(self, idx) -> (torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor):
        img = self.imgs[idx].unsqueeze(0)  # add channel dimension
        target_lm = self.lms[idx]
        dist_map = self.dist_map[idx]
        seg_mask = self.seg_masks[idx]

        return img, target_lm, dist_map, seg_mask

    @classmethod
    def get_training_shapes(cls) -> torch.Tensor:
        return cls('train').lms

    @classmethod
    def get_anatomical_structure_index(cls) -> OrderedDict:
        name_idx_dict = OrderedDict()
        idx = 0
        for organ, num_lms in cls.NUM_LANDMARKS.items():
            name_idx_dict[organ] = (idx, idx + num_lms)
            idx += num_lms

        return name_idx_dict


# quick and dirty buffer for training shapes (preventing multiple loading)
TRAINING_SHAPES = JSRTDataset.get_training_shapes()


class JSRTDatasetUV(JSRTDataset):
    def __init__(self, mode: str):
        super().__init__(mode, normalize_landmarks=False)
        self.uv_maps = torch.load('dataset/data/uv_maps.pth')

        # select for training or testing
        if mode == 'train':
            self.uv_maps = self.uv_maps[:self.SPLIT_IDX]
        elif mode == 'test':
            self.uv_maps = self.uv_maps[self.SPLIT_IDX:]
        else:
            raise ValueError(f'Unknown mode {mode}')

        assert len(self.uv_maps) == super().__len__()

    def __getitem__(self, idx):
        img, target_lm, dist_map, seg_mask = super().__getitem__(idx)
        uv_map = self.uv_maps[idx]

        return img, target_lm, dist_map, seg_mask, uv_map

    @classmethod
    def get_anatomical_structure_uv_values(cls) -> OrderedDict:
        uv_values = torch.load('dataset/data/mean_shape_uv_values.pth')
        name_uv_dict = OrderedDict()
        for organ in cls.NUM_LANDMARKS.keys():
            name_uv_dict[organ] = uv_values[organ]

        return name_uv_dict


if __name__ == '__main__':
    from matplotlib import pyplot as plt

    train_lms = JSRTDataset.get_training_shapes()
    ds = JSRTDatasetUV('train')
    img, lms, dist_map, seg_mask, uv_map = ds[6]

    plt.figure('Landmarks')
    plt.imshow(img.squeeze(), cmap='gray')
    plt.scatter(lms[:, 0], lms[:, 1], c='r')

    for i, (organ, (start_idx, end_idx)) in enumerate(JSRTDataset.get_anatomical_structure_index().items(), 1):
        fig, ax = plt.subplots(1, 3)
        fig.suptitle(organ)
        ax[0].imshow(seg_mask[i - 1])
        ax[0].scatter(lms[start_idx:end_idx, 0], lms[start_idx:end_idx, 1], c='r')
        ax[0].set_title('Segmentation mask')

        ax[1].imshow(uv_map[i - 1, 0])
        ax[1].scatter(lms[start_idx:end_idx, 0], lms[start_idx:end_idx, 1], c='r')
        ax[1].set_title('U map')

        ax[2].imshow(uv_map[i - 1, 1])
        ax[2].scatter(lms[start_idx:end_idx, 0], lms[start_idx:end_idx, 1], c='r')
        ax[2].set_title('V map')

    plt.show()

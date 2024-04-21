from dataset.jsrt_dataset import JSRTDatasetUV
from kornia.augmentation import AugmentationSequential, RandomAffine
from kornia.constants import DataKey, SamplePadding, Resample
from matplotlib import pyplot as plt
from kornia.geometry.transform import warp_affine, Affine

ds = JSRTDatasetUV('train', 'cartesian')
img, lm, _, seg_mask, uv_map = ds[0]

aug = AugmentationSequential(
    RandomAffine(degrees=15.0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
    data_keys=['image', 'mask', 'image', 'keypoints']
)

aug_img, aug_mask, aug_uv, aug_lm = aug(img.unsqueeze(0), seg_mask.unsqueeze(0), uv_map[:, 0].unsqueeze(0), lm.unsqueeze(0))

fig, ax = plt.subplots(3, 2, figsize=(10, 5))
ax[0, 0].imshow(img.squeeze(), cmap='gray')
ax[0, 0].scatter(lm[:, 0], lm[:, 1])
ax[0, 1].imshow(aug_img.squeeze(), cmap='gray')
ax[0, 1].scatter(aug_lm[0, :, 0], aug_lm[0, :, 1])

ax[1, 0].imshow(seg_mask[0])
ax[1, 1].imshow(aug_mask[0, 0])

ax[2, 0].imshow(uv_map[0, 0])
ax[2, 1].imshow(aug_uv[0, 0])

plt.show()

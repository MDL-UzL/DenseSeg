from matplotlib import pyplot as plt
import h5py
from random import choice

graz_img_seg_lms_storage = h5py.File('/home/ron/Documents/DenseSeg/dataset/data/graz/graz_img_seg_lms.h5', 'r')
keys = list(graz_img_seg_lms_storage.keys())
# random key
key = choice(keys)
storage = graz_img_seg_lms_storage[key]

kpt_idx = 0
for i, bone in enumerate(graz_img_seg_lms_storage.attrs['BONE_LABEL']):
    fig, axs = plt.subplots(1, 2)
    fig.suptitle(bone)
    axs[0].imshow(storage['img'], cmap='gray')
    axs[0].imshow(storage['seg'][i], alpha=storage['seg'][i].astype(float))

    offset = graz_img_seg_lms_storage.attrs['NUM_LMS'][i]
    kpts = storage['lms'][kpt_idx:kpt_idx+offset]
    kpt_idx += offset

    axs[0].scatter(kpts[:, 0], kpts[:, 1], cmap='tab20', c=range(offset))

    axs[1].imshow(storage['dist_map'][i])
    axs[1].scatter(kpts[:, 0], kpts[:, 1], c='r')

plt.show()

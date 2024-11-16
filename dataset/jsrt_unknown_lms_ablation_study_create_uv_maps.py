import torch
from matplotlib import pyplot as plt

from dataset.jsrt_dataset import JSRTDataset
from utils import extract_warped_uv_maps

ds = JSRTDataset('all', normalize_landmarks=False)
mean_shape = JSRTDataset('train', normalize_landmarks=False).lms.mean(dim=0)
shapes = ds.lms
segs = ds.seg_masks.bool()

lms2study = {
    'clavicle': (158, 162),
    'center': 'left_lung',
    'contour': [54, 55],
    'corner': [63, 67]
}
seg_idx = list(ds.NUM_LANDMARKS.keys()).index(lms2study['center'])

unknown_lms_uv = {}
tre = {study_name: [] for study_name in lms2study.keys()}

for i, (study_name, idx_unknown) in enumerate(lms2study.items()):
    if isinstance(idx_unknown, tuple):
        idx_unknown = slice(*idx_unknown)
    elif isinstance(idx_unknown, str):
        idx_unknown = ds.get_anatomical_structure_index()[idx_unknown]
        idx_unknown = slice(*idx_unknown)

    selected_lms_mean = mean_shape[idx_unknown]
    selected_lms = shapes[:, idx_unknown]
    if study_name == 'center' or isinstance(idx_unknown, list):
        selected_lms = selected_lms.mean(dim=1, keepdim=True)
        selected_lms_mean = selected_lms_mean.mean(0, keepdim=True)
    N = selected_lms.shape[1]

    # tre of mean shape (only test split)
    tre[study_name].append(
        torch.linalg.vector_norm(selected_lms - selected_lms_mean, dim=-1, ord=2)[ds.SPLIT_IDX:].mean(1))

    for i, (lms, seg) in enumerate(zip(selected_lms, segs)):
        seg = seg[seg_idx]
        if i == ds.SPLIT_IDX:
            plt.imshow(seg)
            plt.scatter(lms[:, 0], lms[:, 1], c='r', label='ground truth')
            plt.scatter(selected_lms_mean[:, 0], selected_lms_mean[:, 1], c='b', label='mean shape')
            plt.title(f"{i}/{len(segs)}")
            plt.legend()
            plt.show()

    anatomy = slice(*ds.get_anatomical_structure_index()[lms2study['center']])
    _, uv_values = extract_warped_uv_maps(
        mean_shape=torch.cat([selected_lms_mean, mean_shape[anatomy]], dim=0),
        landmarks=torch.cat([selected_lms, shapes[:, anatomy]], dim=1),
        segmentation=segs[:, seg_idx]
    )
    unknown_lms_uv[study_name] = {
        'lms': selected_lms,
        'uv_values': uv_values[:N]
    }

# save uv values
# torch.save(unknown_lms_uv, 'dataset/data/ablation_study_unknown_lms_uv.pth')

# tre mean shape on test split
for study_name, values in tre.items():
    values = torch.cat(values, 0) * ds.PIXEL_RESOLUTION_MM
    print(f'{study_name}: {values.mean().item():.2f} ± {values.std().item():.2f} mm')

# plotting
plt.scatter(mean_shape[:, 0], mean_shape[:, 1], c='r', label='mean shape')
# plot landmark indices
for idx, (x, y) in enumerate(mean_shape):
    plt.text(x, y, str(idx), fontsize=8)
plt.gca().invert_yaxis()
plt.show()

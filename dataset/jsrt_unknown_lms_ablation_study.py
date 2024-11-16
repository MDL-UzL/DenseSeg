import torch
from matplotlib import pyplot as plt

from dataset.jsrt_dataset import JSRTDataset
from utils import extract_warped_uv_maps

ds = JSRTDataset('all', normalize_landmarks=False)
mean_shape = JSRTDataset('train', normalize_landmarks=False).lms.mean(dim=0)
shapes = ds.lms
segs = ds.seg_masks.bool()

unknown_lms_uv = {}
tre_clv = []
tre_center = []

for a_i, (anatomy, idx_unknown) in enumerate(zip(['right_lung', 'left_lung'], [(135, 139), (158, 162)])):
    selected_lms_mean = mean_shape[idx_unknown[0]:idx_unknown[1]]
    selected_lms = shapes[:, idx_unknown[0]:idx_unknown[1]]
    N = selected_lms.shape[1]

    # add center
    start_idx, end_idx = JSRTDataset.get_anatomical_structure_index()[anatomy]
    centers = shapes[:, start_idx:end_idx].mean(dim=1)
    center_mean_shape = mean_shape[start_idx:end_idx].mean(0)

    # tre of mean shape (only test split)
    tre_clv.append(
        torch.linalg.vector_norm(selected_lms - selected_lms_mean.unsqueeze(0), dim=-1, ord=2)[ds.SPLIT_IDX:].mean(1))
    tre_center.append(torch.linalg.vector_norm(centers - center_mean_shape.unsqueeze(0), dim=-1, ord=2)[ds.SPLIT_IDX:])

    for i, (lms, center, seg) in enumerate(zip(selected_lms, centers, segs)):
        seg = seg[a_i]
        lms = lms.round().int()
        center = center.round().int()
        if not seg[lms[:, 1], lms[:, 0]].all() or not seg[center[1], center[0]]:
            print("Not all landmarks are inside the segmentation mask and hence would have")
            plt.imshow(seg)
            plt.scatter(lms[:, 0], lms[:, 1], c='r')
            plt.scatter(center[0], center[1], c='b')
            plt.title(f"{i}/{N}")

    _, uv_values = extract_warped_uv_maps(
        mean_shape=torch.cat([mean_shape[start_idx:end_idx], selected_lms_mean, center_mean_shape.unsqueeze(0)], dim=0),
        landmarks=torch.cat([shapes[:, start_idx:end_idx], selected_lms, centers.unsqueeze(1)], dim=1),
        segmentation=segs[:, a_i]
    )
    unknown_lms_uv[anatomy] = {
        'clavicles': {'lms': selected_lms, 'uv_values': uv_values[-(N + 1):-1]},
        'center': {'lms': centers.unsqueeze(1), 'uv_values': uv_values[-1].unsqueeze(0)}
    }

# tre mean shape on test split
tre_clv = torch.cat(tre_clv, 0) * ds.PIXEL_RESOLUTION_MM
tre_center = torch.cat(tre_center, 0) * ds.PIXEL_RESOLUTION_MM
print(f'TRE clavicles: {tre_clv.mean().item()} +- {tre_clv.std().item()}')
print(f'TRE center: {tre_center.mean().item()} +- {tre_center.std().item()}')

# save uv values
# torch.save(unknown_lms_uv, 'dataset/data/ablation_study_unknown_lms_uv.pth')

# plotting
plt.scatter(mean_shape[:, 0], mean_shape[:, 1], c='r', label='mean shape')
# plot landamrk indices
for idx, (x, y) in enumerate(mean_shape):
    plt.text(x, y, str(idx), fontsize=8)
plt.scatter(selected_lms_mean[:, 0], selected_lms_mean[:, 1], c='k', marker='x', label='selected landmarks')
plt.gca().invert_yaxis()
plt.show()

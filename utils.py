import numpy as np
import torch
from scipy.interpolate import griddata
from tqdm import tqdm
from torch.nn import functional as F
from kornia.geometry import warp_affine, invert_affine_transform
from torchvision.transforms.functional import crop


# https://github.com/clementinboittiaux/umeyama-python/blob/main/umeyama.py
def umeyama(X, Y):
    """
    Estimates the Sim(3) transformation between `X` and `Y` point sets.

    Estimates c, R and t such as c * R @ X + t ~ Y.

    Parameters
    ----------
    X : numpy.array
        (m, n) shaped numpy array. m is the dimension of the points,
        n is the number of points in the point set.
    Y : numpy.array
        (m, n) shaped numpy array. Indexes should be consistent with `X`.
        That is, Y[:, i] must be the point corresponding to X[:, i].

    Returns
    -------
    c : float
        Scale factor.
    R : numpy.array
        (3, 3) shaped rotation matrix.
    t : numpy.array
        (3, 1) shaped translation vector.
    """
    mu_x = X.mean(axis=1).reshape(-1, 1)
    mu_y = Y.mean(axis=1).reshape(-1, 1)
    var_x = np.square(X - mu_x).sum(axis=0).mean()
    cov_xy = ((Y - mu_y) @ (X - mu_x).T) / X.shape[1]
    U, D, VH = np.linalg.svd(cov_xy)
    S = np.eye(X.shape[0])
    if np.linalg.det(U) * np.linalg.det(VH) < 0:
        S[-1, -1] = -1
    c = np.trace(np.diag(D) @ S) / var_x
    R = U @ S @ VH
    t = mu_y - c * R @ mu_x
    return c, R, t


def apply_umeyama_on_tensor(X: torch.Tensor, Y: torch.Tensor):
    """
    Applies the umeyama transformation on the tensor `X` to align it with `Y`.
    :param X: (n, d) shaped tensor. n is the number of points, d is the dimension of the points.
    :param Y: (n, d) shaped tensor. Indexes should be consistent with `X`.
        That is, Y[i] must be the point corresponding to X[i].
    :return: version of `X` aligned with `Y` and the transformation matrix (d, d+1).
    """
    X = X.cpu().numpy()
    Y = Y.cpu().numpy()
    c, R, t = umeyama(X.T, Y.T)
    # theta is transformed to work with the transposed tensors
    theta = torch.cat([c * torch.from_numpy(R).float().t(), torch.from_numpy(t).float().flip(0)], dim=1)
    return torch.tensor(c * R @ X.T + t).T, theta


def extract_warped_uv_maps(mean_shape: torch.Tensor, landmarks: torch.Tensor, segmentation: torch.Tensor) -> tuple:
    """
    Generate the UV map for the mean shape and warp it to the shape of the landmarks. The returned UV maps are masked,
    but could still contain NaN values. All landmarks must be given in pixel coordinates.
    :param mean_shape: mean shape of the object of shape (L, 2)
    :param landmarks: landmarks of the objects of shape (N, L, 2)
    :param segmentation: segmentation masks of the objects of shape (N, H, W)
    :return: warped UV maps of the objects of shape (N, 2, H, W), mean shape's UV values of shape (L, 2)
    """

    N, L, _ = landmarks.shape
    _, H, W = segmentation.shape
    assert mean_shape.shape[-1] == landmarks.shape[-1] == 2, 'Landmarks must be 2D'
    assert mean_shape.shape[0] == L, 'Number of landmarks must match the mean shape'
    assert segmentation.shape[0] == N, 'Number of samples must match the segmentation'
    assert (mean_shape[:, 0].min() >= 0 and mean_shape[:, 0].max() <= W and
            mean_shape[:, 1].min() >= 0 and mean_shape[:, 1].max() <= H), 'Mean shape must be within the image bounds'
    assert (landmarks[:, :, 0].min() >= 0 and landmarks[:, :, 0].max() <= W and
            landmarks[:, :, 1].min() >= 0 and landmarks[:, :,
                                              1].max() <= H), 'Landmarks must be within the image bounds'

    uv_maps = torch.empty((N, 2, H, W), dtype=torch.float32)
    atlas = mean_shape
    # create identity uv map
    range_atlas = torch.max(atlas, dim=0).values - torch.min(atlas, dim=0).values
    range_atlas_int = range_atlas.floor().int()
    uv = torch.meshgrid(
        torch.linspace(-1, 1, range_atlas_int[0]),
        torch.linspace(-1, 1, range_atlas_int[1]),
        indexing='ij'
    )
    uv = torch.stack(uv, dim=0)

    # extract atlas's uv values
    atlas_grid_sample = (atlas - torch.min(atlas, dim=0).values) / range_atlas  # [0, 1]
    atlas_grid_sample = atlas_grid_sample * 2 - 1  # [-1, 1]

    atlas_uv_values = F.grid_sample(uv.unsqueeze(0), atlas_grid_sample.view(1, 1, -1, 2).flip(-1), mode='bilinear',
                                    padding_mode='zeros', align_corners=True).squeeze().transpose(0, 1)
    # flip to make vu → uv
    atlas_uv_values = atlas_uv_values.flip(-1)

    for n, (sample, seg_mask) in enumerate(tqdm(zip(landmarks, segmentation), total=N, desc='Generating UV maps')):
        # Prealignment
        fixed, theta = apply_umeyama_on_tensor(sample, atlas)

        # Normalization to [-1, 1]
        pc = torch.cat([atlas, fixed], dim=0)
        range_pc = torch.max(pc, dim=0).values - torch.min(pc, dim=0).values

        fixed_norm = (fixed - torch.min(pc, dim=0).values) / range_pc.max()  # bigger axis [0, 1], smaller axis [0, <1]
        fixed_norm = fixed_norm * 2 - (
                range_pc / range_pc.max())  # bigger axis [-1, 1], smaller axis [>-1, <1] (both centered around 0)

        atlas_norm = (atlas - torch.min(pc, dim=0).values) / range_pc.max()
        atlas_norm = atlas_norm * 2 - (range_pc / range_pc.max())

        pc_norm = torch.cat([atlas_norm, fixed_norm], dim=0)

        # Registration
        displacement = fixed_norm - atlas_norm

        # Interpolate to dense displacement field
        # bounding box
        bb_min = torch.min(pc_norm, dim=0).values
        bb_max = torch.max(pc_norm, dim=0).values
        range_pc_int = range_pc.floor().int()

        # create grid for bounding box keeping the same amount of "pixel" as in the image
        grid = torch.meshgrid(torch.linspace(bb_min[0], bb_max[0], range_pc_int[0]),
                              torch.linspace(bb_min[1], bb_max[1], range_pc_int[1]), indexing='ij')
        grid = torch.stack(grid, dim=-1).reshape(-1, 2)

        # bilinear interpolation
        grid_displacement = griddata(fixed_norm, -displacement, grid, method='linear', rescale=True)
        grid_displacement = torch.from_numpy(grid_displacement).float()

        # finding the starting point of the atlas in the grid and calculate the bounding box
        atlas_grid_start = torch.norm(grid.view(-1, 2) - atlas_norm.min(0).values, dim=-1).argmin()
        atlas_grid_start = torch.tensor([atlas_grid_start // range_pc_int[1], atlas_grid_start % range_pc_int[1]])
        atlas_grid_end = atlas_grid_start + range_atlas_int

        # crop the uv if it is outside the grid
        grid_space_overshot = range_atlas_int - range_pc_int + atlas_grid_start
        grid_space_overshot = torch.clamp_min(grid_space_overshot, 0)
        uv_cropped_H, uv_cropped_W = range_atlas_int - grid_space_overshot
        cropped_uv = crop(uv, 0, 0, uv_cropped_H, uv_cropped_W)

        # place uv map in the grid
        uv_grid = torch.full((2, range_pc_int[0], range_pc_int[1]), torch.nan)
        uv_grid[:, atlas_grid_start[0]:atlas_grid_end[0], atlas_grid_start[1]:atlas_grid_end[1]] = cropped_uv

        # warp uv map to fixed
        warped_grid = grid + grid_displacement
        warped_uv_grid = F.grid_sample(uv_grid.unsqueeze(0),
                                       warped_grid.view(1, range_pc_int[0], range_pc_int[1], 2).flip(-1),
                                       mode='bilinear', align_corners=True).squeeze(0)

        # placed warped uv map back into original image space
        img_size = seg_mask.shape[::-1]
        # find bounding box of the point cloud in the image space
        pc_bbox_min = torch.min(pc, dim=0).values.round().int()
        pc_bbox_max = torch.tensor(warped_uv_grid.shape[-2:]) + pc_bbox_min

        uv_img_space = torch.full((2, img_size[0], img_size[1]), torch.nan)
        uv_img_space[:, pc_bbox_min[0]:pc_bbox_max[0], pc_bbox_min[1]:pc_bbox_max[1]] = warped_uv_grid

        # reverse the prealignment to put the warped uv map of fixed to position of sample
        theta_23 = theta[:2, :].unsqueeze(0)
        theta_23_inv = invert_affine_transform(theta_23)

        uv_img_space_corrected = warp_affine(uv_img_space.unsqueeze(0), theta_23_inv, dsize=(img_size[0], img_size[1]),
                                             align_corners=True).squeeze()

        # mask the uv map to the object boundaries utilizing the segmentation mask
        uv_img_space_masked = uv_img_space_corrected.transpose(1, 2)
        uv_img_space_masked[:, seg_mask.logical_not()] = torch.nan

        # flip to make vu → uv
        uv_img_space_masked = uv_img_space_masked.flip(0)

        # save the uv map
        uv_maps[n] = uv_img_space_masked

    return uv_maps, atlas_uv_values


def convert_uv_to_coordinates(uv_map: torch.Tensor, uv_values: torch.Tensor, mode: str, k: int = None) -> torch.Tensor:
    """
    Calculate the coordinates of the uv_values in the uv_map by finding the closest uv value in the uv_map.
    UV maps may contain NaN values, which are ignored in the calculation.
    :param uv_map: UV map of shape (B, 2, H, W)
    :param uv_values: UV values of shape (B, N_c, 2) where N_c is the number of uv values for each class C
    :param mode: 'nearest' or 'linear'
    :param k: number of closest points to consider in linear mode
    :return: coordinates of the uv_values in the uv_map of shape (B, N_c, 2) where N_c is the number of uv values for each class C
    """

    assert uv_map.shape[1] == 2, 'UV map must have 2 channels'
    assert uv_values.shape[-1] == 2, 'UV values must have 2 dimensions'
    assert uv_map.shape[0] == uv_values.shape[0], 'Batch size of uv_map and uv_values must match'
    assert k is None or (k > 0 and isinstance(k, int)), 'k must be a positive integer'
    assert uv_map.isnan().logical_not().sum() >= k, 'UV map must contain at least k valid values'

    B, _, H, W = uv_map.shape
    device = uv_map.device

    # calculate the distance of each uv value to each uv coordinate
    uv_coord_dist = uv_map.view(B, 2, H * W, 1) - uv_values.transpose(1, 2).view(B, 2, 1, -1)  # (B, 2, H*W, N)
    uv_coord_dist = torch.linalg.vector_norm(uv_coord_dist, dim=1)  # (B, H*W, N)

    # extract uv coordinate
    if mode == 'nearest' or k == 1:
        uv_coord = nanargmin(uv_coord_dist, dim=1)  # (B, N)
        uv_coord = torch.stack([uv_coord // W, uv_coord % W], dim=-1)  # (B, N, 2)
    elif mode == 'linear':
        assert k is not None, 'k must be given in linear mode'
        uv_coord = torch.arange(H * W, device=device)
        uv_coord = torch.stack([uv_coord // W, uv_coord % W], dim=-1)  # (H*W, 2)
        uv_coord = uv_coord.unsqueeze(0).expand(B, H * W, -1)  # (B, H*W, 2)

        # select the k closest points
        top_k_indices = torch.topk(uv_coord_dist, k=k, dim=1, largest=False).indices  # (B, H*W, k)
        top_k_mask = torch.ones_like(uv_coord_dist, dtype=torch.bool)  # (B, H*W, N)
        top_k_mask.scatter_(1, top_k_indices, False)

        # replace non-top k values with -infinity due to its neutral behavior in the softmax (softmin: inf → -inf)
        # NaN values are already implicitly covered by top k
        uv_coord_dist = torch.where(top_k_mask, torch.inf, uv_coord_dist)
        weights = F.softmin(uv_coord_dist, dim=1)  # (B, H*W, N)
        # (B, H*W, 1, 2) * (B, H*W, N, 1) → (B, H*W, N, 2) → (B, N, 2)
        uv_coord = torch.sum(uv_coord.unsqueeze(-2) * weights.unsqueeze(-1), dim=1)
    else:
        raise ValueError(f'Unknown mode: {mode}. Has to be "nearest" or "linear".')

    # HW → WH
    uv_coord = uv_coord.flip(-1)

    return uv_coord


def convert_list_of_uv_to_coordinates(uv_map: torch.Tensor, uv_values: list, mode: str, k: int = None) -> list:
    """
    Calculate the coordinates of the uv_values in the uv_map by finding the closest uv value in the uv_map for each class C.
    UV maps may contain NaN values, which are ignored in the calculation.
    :param uv_map: uv map of shape (B, C, 2, H, W)
    :param uv_values: list of length C containing uv values of shape (N, 2)
    :param mode: 'nearest' or 'linear'
    :param k: number of closest points to consider in linear mode
    :return: list of length C containing coordinates of the uv_values in the uv_map of shape (B, N, 2)
    """
    B, C = uv_map.shape[:2]
    assert len(uv_values) == C, 'Number of list entries must match the number of classes'
    # handle each class independently due to different number of landmarks and different uv maps
    uv_values_B = list(map(lambda uv: uv.unsqueeze(0).expand(B, -1, -1), uv_values))
    coord_list = list(map(lambda c: convert_uv_to_coordinates(uv_map[:, c], uv_values_B[c], mode, k), range(C)))

    return coord_list


def nanargmin(tensor, dim=None, keepdim=False):
    max_value = torch.finfo(tensor.dtype).max
    output = tensor.nan_to_num(max_value).argmin(dim=dim, keepdim=keepdim)
    return output



if __name__ == '__main__':
    from dataset import JSRTDataset, TRAINING_SHAPES

    from matplotlib import pyplot as plt

    anatomy = 'right_clavicle'
    anatomy_idx = JSRTDataset.get_anatomical_structure_index()[anatomy]
    shapes = (TRAINING_SHAPES[:, anatomy_idx[0]:anatomy_idx[1]] + 1) / 2 * 256
    mean_shape = shapes.mean(0)
    seg_masks = JSRTDataset('train').seg_masks[:,
                list(JSRTDataset.get_anatomical_structure_index().keys()).index(anatomy)].bool()

    uv_maps = extract_warped_uv_maps(mean_shape, shapes, seg_masks)

    rnd_idx = np.random.randint(0, len(uv_maps) - 1)
    fig, axs = plt.subplots(1, 3)
    axs[0].imshow(uv_maps[rnd_idx, 0])
    # axs[0].scatter(mean_shape[:, 0], mean_shape[:, 1], c='r')
    axs[1].imshow(uv_maps[rnd_idx, 1])
    axs[2].imshow(seg_masks[rnd_idx], cmap='gray')
    plt.show()

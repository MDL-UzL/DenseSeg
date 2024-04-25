import h5py
import torch
from clearml import InputModel
from tqdm import tqdm

from clearml_ids import grazer_model_ids
from models.uv_unet import UVUNet
from utils import convert_list_of_uv_to_coordinates

device = 'cuda:3' if torch.cuda.is_available() else 'cpu'

cl_model = InputModel(grazer_model_ids['uv'])
model = UVUNet.load(cl_model.get_weights(), 'cpu').eval().to(device)

graz_img_seg_lms_storage = h5py.File('dataset/data/graz/graz_img_seg_lms800.h5', 'r')
result_storage = h5py.File('dataset/data/graz/graz_prediction_800split.h5', 'w')

uv_values = torch.load('dataset/data/graz/mean_shape_uv_values_polar.pth')
uv_values = [v.to(device, non_blocking=True) for v in uv_values.values()]
with torch.inference_mode():
    for file in tqdm(graz_img_seg_lms_storage.keys(), desc='Evaluating', unit='img',
                     total=len(graz_img_seg_lms_storage.keys())):
        img = torch.from_numpy(graz_img_seg_lms_storage[file]['img'][:]).float().to(device)

        seg_hat, uv_hat = model.predict(img.unsqueeze(0).unsqueeze(0), mask_uv=True)
        seg_hat = seg_hat.squeeze(0)
        try:
            lm_hat = convert_list_of_uv_to_coordinates(uv_hat, uv_values, 'linear', k=5)
        except AssertionError:
            print(f'Skip {file}')
            continue
        lm_hat = torch.cat(lm_hat, dim=1).squeeze(0)

        grp = result_storage.create_group(file)
        grp.create_dataset('seg', data=seg_hat.cpu().numpy())
        grp.create_dataset('lms', data=lm_hat.cpu().numpy())

        # from matplotlib import pyplot as plt
        # plt.imshow(img.squeeze(), cmap='gray')
        # plt.scatter(lm_hat[:, 0], lm_hat[:, 1], c='r')
        #
        # plt.figure()
        # plt.imshow(img.squeeze(), cmap='gray')
        # plt.imshow(seg_hat.float().argmax(0), alpha=seg_hat.any(0).float())
        # plt.show()

graz_img_seg_lms_storage.close()

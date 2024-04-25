import h5py
import torch
from clearml import InputModel
from tqdm import tqdm

from clearml_ids import grazer_model_ids
from models.kpts_unet import KeypointUNet
from utils import extract_kpts_from_heatmap

device = 'cuda:3' if torch.cuda.is_available() else 'cpu'

cl_model = InputModel(grazer_model_ids['heatmap'])
model = KeypointUNet.load(cl_model.get_weights(), 'cpu').eval().to(device)

graz_img_seg_lms_storage = h5py.File('dataset/data/graz/graz_img_seg_lms800.h5', 'r')
result_storage = h5py.File('dataset/data/graz/graz_prediction_800split_heat_reg.h5', 'w')

with torch.inference_mode():
    for file in tqdm(graz_img_seg_lms_storage.keys(), desc='Evaluating', unit='img',
                     total=len(graz_img_seg_lms_storage.keys())):
        img = torch.from_numpy(graz_img_seg_lms_storage[file]['img'][:]).float().to(device)

        heatmap_hat = model(img.unsqueeze(0).unsqueeze(0))
        lm_hat = extract_kpts_from_heatmap(heatmap_hat).squeeze(0)

        grp = result_storage.create_group(file)
        grp.create_dataset('lms', data=lm_hat.cpu().numpy())

        # from matplotlib import pyplot as plt
        # plt.imshow(img.squeeze(), cmap='gray')
        # plt.scatter(lm_hat[:, 0], lm_hat[:, 1], c='r')
        # plt.show()

graz_img_seg_lms_storage.close()

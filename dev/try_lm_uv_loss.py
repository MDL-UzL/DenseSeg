from training.forward_func import landmark_uv_loss
from dataset.jsrt_dataset import JSRTDatasetUV
from torch import nn

img, target_lm, _, _, uv_map = JSRTDatasetUV('train')[0]
lm_uv_values = JSRTDatasetUV.get_anatomical_structure_uv_values()
lm_uv_values = [uv_values for uv_values in lm_uv_values.values()]

uv_map = uv_map.nan_to_num(0)

target_lm[:20] = 300

loss = landmark_uv_loss(uv_map.unsqueeze(0), target_lm.unsqueeze(0), lm_uv_values, nn.L1Loss(reduction='none'))

import os
from tempfile import gettempdir

import torch
from clearml import Task
from torch.utils.data import DataLoader
from tqdm import trange

from dataset.jsrt_dataset import JSRTDatasetUV
from models.uv_unet import UVUNet
from training.forward_func import forward
from training.hyper_params import hp_parser
from kornia.augmentation import AugmentationSequential, RandomAffine

hp = hp_parser.parse_args()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

if hp.bce and any([hp.reg_uv]):
    task_name = 'Segmentation and UV Map'
    tags = [hp.uv_loss if hp.reg_uv else '', 'TV' if hp.tv else '']
elif hp.bce:
    task_name = 'Segmentation'
    tags = []
elif hp.reg_uv:
    task_name = f'UV Map {hp.uv_loss}'
    tags = [hp.uv_loss, 'TV' if hp.tv else '']
else:
    raise ValueError('At least one of seg or uv must be True')
tags.append(hp.uv_method)
tags.append(hp.supervision)

use_data_aug = hp.rotate or hp.translate or hp.scale
if use_data_aug:
    tags.append('DataAug')

task = Task.init(project_name='DenseSeg', task_name=task_name, tags=tags, auto_connect_frameworks=False,
                 auto_connect_arg_parser={'gpu_id': False}, auto_resource_monitoring=False)
# init pytorch
torch.manual_seed(hp.seed)
if hp.gpu_id is None and torch.cuda.is_available():  # workaround to enable GPU selection during HPO
    hp.gpu_id = 0
    Warning('GPU is available but not selected. Defaulting to GPU 0 to enable CUDA_VISIBLE_DEVICES selection.')
device = torch.device(f'cuda:{hp.gpu_id}' if torch.cuda.is_available() else 'cpu')

# define data loaders
dl_kwargs = {'num_workers': 4, 'pin_memory': True} if torch.cuda.is_available() else {}
train_dl = DataLoader(JSRTDatasetUV('train', hp.uv_method), batch_size=hp.batch_size, drop_last=True, **dl_kwargs)
val_dl = DataLoader(JSRTDatasetUV('test', hp.uv_method), batch_size=hp.infer_batch_size, shuffle=False, drop_last=False,
                    **dl_kwargs)

# define model
n_classes = train_dl.dataset.N_CLASSES
model = UVUNet(n_classes)
model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=hp.lr)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=hp.epochs, eta_min=hp.lr / 100)

if hp.uv_loss == 'l2':
    uv_loss_fn = torch.nn.MSELoss(reduction='none')
elif hp.uv_loss == 'l1':
    uv_loss_fn = torch.nn.L1Loss(reduction='none')
elif hp.uv_loss == 'smoothl1':
    uv_loss_fn = torch.nn.SmoothL1Loss(reduction='none', beta=hp.beta)
else:
    raise ValueError(f'Unknown uv loss function {hp.uv_loss}')
lm_uv_values = [uv_values.to(device) for uv_values in train_dl.dataset.get_anatomical_structure_uv_values().values()]

if use_data_aug:
    data_aug = AugmentationSequential(
        RandomAffine(degrees=hp.rotate, translate=(hp.translate,) * 2, scale=(1 - hp.scale, 1 + hp.scale), p=1),
        data_keys=['image', 'mask', 'image', 'image', 'keypoints']
    )
else:
    data_aug = None

fwd_kwargs = {'model': model, 'optimizer': optimizer, 'device': device, 'lambdas': [hp.bce, hp.reg_uv, hp.tv],
              'lm_uv_values': lm_uv_values, 'uv_loss_fn': uv_loss_fn, 'data_aug': data_aug,
              'supervision': hp.supervision, 'bce_pos_weight': train_dl.dataset.BCE_POS_WEIGHTS.to(device)}

for epoch in trange(hp.epochs, desc='training'):
    forward('train', train_dl, epoch, **fwd_kwargs)
    forward('val', val_dl, epoch, **fwd_kwargs)

    if hp.lr_scheduler:
        scheduler.step()
        # log learning rate
        task.get_logger().report_scalar(title='Learning rate', series='lr', value=scheduler.get_last_lr()[0],
                                        iteration=epoch)

# save model to ClearML
save_path = gettempdir() + '/uvunet.pth'
model.save(save_path)
task.update_output_model(save_path, model_name='final_model')
task.close()

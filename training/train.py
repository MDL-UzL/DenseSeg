from tempfile import gettempdir

import torch
from clearml import Task
from torch.utils.data import DataLoader
from tqdm import trange

from dataset.jsrt_dataset import JSRTDatasetUV
from models.uv_unet import UVUNet
from training.forward_func import forward
from training.hyper_params import hp_parser

hp = hp_parser.parse_args()

if hp.seg and hp.uv:
    task_name = 'Segmentation and UV Map'
    tags = [hp.uv_loss]
elif hp.seg:
    task_name = 'Segmentation'
    tags = []
elif hp.uv:
    task_name = f'UV Map {hp.uv_loss}'
    tags = [hp.uv_loss]
else:
    raise ValueError('At least one of seg or uv must be True')

task = Task.init(project_name='DenseSeg', task_name=task_name, auto_connect_frameworks=False, tags=tags)
# init pytorch
torch.manual_seed(hp.seed)
device = torch.device(f'cuda:{hp.gpu_id}' if torch.cuda.is_available() else 'cpu')

# define data loaders
dl_kwargs = {'num_workers': 4, 'pin_memory': True} if torch.cuda.is_available() else {}
train_dl = DataLoader(JSRTDatasetUV('train'), batch_size=hp.batch_size, drop_last=True, **dl_kwargs)
val_dl = DataLoader(JSRTDatasetUV('test'), batch_size=hp.infer_batch_size, shuffle=False, drop_last=False, **dl_kwargs)

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

fwd_kwargs = {'model': model, 'optimizer': optimizer, 'device': device, 'train_seg': hp.seg, 'train_uv': hp.uv,
              'bce_pos_weight': train_dl.dataset.BCE_POS_WEIGHTS.to(device), 'uv_loss_fn': uv_loss_fn}

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

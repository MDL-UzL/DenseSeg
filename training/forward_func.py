import torch
from clearml import Logger
from monai import metrics
from torch.nn import functional as F
from torch.nn.modules.loss import _Loss
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from models.uv_unet import UVUNet

torch.autograd.set_detect_anomaly(True)


def balanced_normalized_uv_loss(uv_hat: torch.Tensor, uv: torch.Tensor, loss_fn: _Loss) -> torch.Tensor:
    with torch.no_grad():
        assert loss_fn.reduction == 'none', 'loss_fn must have reduction set to none'
        assert not uv.isnan().all(), 'uv must contain some valid values'
        assert not uv_hat.isnan().all(), 'uv_hat must contain some valid values'

    # (B, C, 2, H, W) -> (B, C, 2 * H * W)
    uv_hat_flat = uv_hat.flatten(start_dim=2)
    uv_flat = uv.flatten(start_dim=2)

    # zero out NaN values in ground truth to prevent NaN loss
    nan_gt_mask = uv_flat.isnan()
    uv_flat = torch.where(nan_gt_mask, 0, uv_flat)
    uv_hat_flat = torch.where(nan_gt_mask, 0, uv_hat_flat)

    # normalize each class with its number of valid pixels
    valid_pxl = nan_gt_mask.logical_not().sum(-1)  # (B, C)
    loss = loss_fn(uv_hat_flat, uv_flat).sum(-1) / valid_pxl  # (B, C)

    return loss

@torch.no_grad()
def uv_l1_loss(uv_hat: torch.Tensor, uv: torch.Tensor) -> torch.Tensor:
    return balanced_normalized_uv_loss(uv_hat, uv, torch.nn.L1Loss(reduction='none'))

def forward(mode: str, data_loader: DataLoader, epoch: int,  # have to be given each call
            # can be provided via kwargs dict
            train_seg: bool, train_uv: bool, model: UVUNet, optimizer: Optimizer, device: torch.device,
            bce_pos_weight: torch.Tensor, uv_loss_fn) -> (torch.Tensor, torch.Tensor):
    assert train_uv or train_seg, 'At least one of train_uv or train_seg must be True'
    # set model mode according to mode
    if mode == 'train':
        model.train()
    elif mode in ['test', 'val']:
        model.eval()
    else:
        raise ValueError(f'Unknown mode: {mode}')

    dsc = metrics.DiceMetric(reduction='mean_batch', include_background=True, ignore_empty=True,
                             num_classes=data_loader.dataset.N_CLASSES)
    uv_l1 = metrics.LossMetric(uv_l1_loss, reduction='mean_batch')
    loss_collector = metrics.CumulativeAverage()

    for img, _, _, seg_mask, uv_map in data_loader:
        img = img.to(device, non_blocking=True)
        seg = seg_mask.to(device, non_blocking=True)
        uv = uv_map.to(device, non_blocking=True)

        with torch.set_grad_enabled(model.training):  # forward
            seg_hat, uv_hat = model(img)
            bce_loss = F.binary_cross_entropy_with_logits(seg_hat, seg, pos_weight=bce_pos_weight) if train_seg else 0
            uv_loss = balanced_normalized_uv_loss(uv_hat, uv, uv_loss_fn).mean() if train_uv else 0
            loss = bce_loss + uv_loss

        if model.training:  # backward
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        # track metrics
        batch_size = len(img)
        loss_collector.append([loss, bce_loss, uv_loss], count=batch_size)
        dsc(seg_hat.sigmoid() > 0.5, seg)
        uv_l1(uv_hat, uv)

    # log metrics scalars
    log = Logger.current_logger()
    loss_avg = loss_collector.aggregate()  # [loss, bce, uv]
    log.report_scalar('Loss', mode, iteration=epoch, value=loss_avg[0].item())
    if train_seg:
        log.report_scalar('BCE', mode, iteration=epoch, value=loss_avg[1].item())
        log.report_scalar('Dice', mode, iteration=epoch, value=dsc.aggregate().mean().item())
        log.report_histogram('Dice', mode, iteration=epoch,
                             values=dsc.aggregate().cpu().numpy(),
                             xlabels=data_loader.dataset.CLASS_LABEL, xaxis='class', yaxis='dice')
    if train_uv:
        log.report_scalar('UV Loss', mode, iteration=epoch, value=loss_avg[2].item())
        log.report_scalar('UV L1', mode, iteration=epoch, value=uv_l1.aggregate().mean().item())
        log.report_histogram('UV L1', mode, iteration=epoch,
                             values=uv_l1.aggregate().cpu().numpy(),
                             xlabels=data_loader.dataset.CLASS_LABEL, xaxis='class', yaxis='uv l1')

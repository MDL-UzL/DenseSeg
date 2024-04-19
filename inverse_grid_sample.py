import torch
from torch.autograd import Function
from torch.autograd.functional import jacobian
from torch.nn import functional as F
from torch import nn

def kpts_pt(kpts_world, shape):
    device = kpts_world.device
    return (kpts_world.flip(-1) / (shape.flip(-1) - 1)) * 2 - 1

def homogenous(kpts):
    B, N, _ = kpts.shape
    device = kpts.device
    return torch.cat([kpts, torch.ones(B, N, 1, device=device)], dim=2)

class InverseGridSampler(Function):
    @staticmethod
    def forward(ctx, input, grid, shape, mode, padding_mode, align_corners):
        ctx.save_for_backward(input, grid)
        ctx.mode = mode
        ctx.padding_mode = padding_mode
        ctx.align_corners = align_corners

        dtype = input.dtype
        device = input.device
        output = -jacobian(
            lambda x: (F.grid_sample(x, grid, mode, padding_mode, align_corners) - input).pow(2).mul(0.5).sum(),
            torch.zeros(shape, dtype=dtype, device=device))

        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, grid = ctx.saved_tensors
        mode = ctx.mode
        padding_mode = ctx.padding_mode
        align_corners = ctx.align_corners

        B, C = input.shape[:2]
        input_shape = input.shape[2:]
        grad_output_shape = grad_output.shape[2:]

        y = jacobian(
            lambda x: F.grid_sample(grad_output.unsqueeze(2).view(B * C, 1, *grad_output_shape), x, mode, padding_mode,
                                    align_corners).mean(),
            grid.unsqueeze(1).repeat(1, C, *([1] * (len(input_shape) + 1))).view(B * C, *input_shape,
                                                                                 len(input_shape))).view(B, C,
                                                                                                         *input_shape,
                                                                                                         len(input_shape))
        grad_grid = (input.numel() * input.unsqueeze(-1) * y).sum(1)

        grad_input = F.grid_sample(grad_output, grid, mode, padding_mode, align_corners)

        return grad_input, grad_grid, None, None, None, None


def inverse_grid_sample(input, grid, shape, mode='bilinear', padding_mode='zeros', align_corners=True):
    return InverseGridSampler.apply(input, grid, shape, mode, padding_mode, align_corners)


def filter1D(img, weight, dim, padding_mode='replicate'):
    B, C = img.shape[:2]
    dims = img.shape[2:]
    N = weight.shape[0]

    padding = torch.zeros(2 * len(dims), )
    padding[[2 * len(dims) - 2 - 2 * dim, 2 * len(dims) - 1 - 2 * dim]] = N // 2
    padding = padding.long().tolist()

    view = torch.ones(2 + len(dims), )
    view[dim + 2] = -1
    view = view.long().tolist()

    if len(dims) == 2:
        return F.conv2d(F.pad(img.view(B * C, 1, *dims), padding, mode=padding_mode), weight.view(view)).view(B, C,
                                                                                                              *dims)

    if len(dims) == 3:
        return F.conv3d(F.pad(img.view(B * C, 1, *dims), padding, mode=padding_mode), weight.view(view)).view(B, C,
                                                                                                              *dims)


class GaussianSmoothing(nn.Module):
    def __init__(self, sigma):
        super(GaussianSmoothing, self).__init__()

        sigma = torch.tensor([sigma])
        N = torch.ceil(sigma * 3.0 / 2.0).long().item() * 2 + 1

        weight = torch.exp(-torch.pow(torch.linspace(-(N // 2), N // 2, N), 2) / (2 * torch.pow(sigma, 2)))
        weight /= weight.sum()

        self.weight = weight

    def forward(self, x):
        D = len(x.shape) - 2
        device = x.device

        for d in range(D):
            x = filter1D(x, self.weight.to(device), d)

        return x


def reconstruct_kpts(kpts, label, shape, scale=1, mode='bilinear', padding_mode='zeros', align_corners=True,
                     normalise=False):
    _, N, C = label.shape
    B, _, D = kpts.shape
    device = kpts.device

    shape_scaled = (shape * scale).round().long()

    kpts_view = [B, *[1 for _ in range(D - 1)], N, D]
    label_view = [B, -1, *[1 for _ in range(D - 1)], N]

    if normalise:
        rec = inverse_grid_sample(homogenous(label).permute(0, 2, 1).view(label_view),
                                  kpts_pt(kpts, shape).view(kpts_view), (B, C + 1, *(shape_scaled)), mode, padding_mode,
                                  align_corners)
        rec = rec[:, :-1] / (rec[:, -1:] + 1e-8)
    else:
        rec = inverse_grid_sample(label.permute(0, 2, 1).view(label_view), kpts_pt(kpts, shape).view(kpts_view),
                                  (B, C, *(shape_scaled)), mode, padding_mode, align_corners)

    return rec
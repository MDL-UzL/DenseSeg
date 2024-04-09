from monai.networks import nets, blocks
from models.modelio import LoadableModel, store_config_args
from torch import nn
import torch


def cat_residual_units(n_units: int, in_channels: int, out_channels: int, norm: str, act: str) -> nn.Sequential:
    layer = []
    for _ in range(n_units):
        layer.append(
            blocks.ResidualUnit(
                spatial_dims=2,
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                norm=(norm, {'affine': True}),
                act=act,
                bias=False
            )
        )
    layer = nn.Sequential(*layer)
    return layer


class UVUNet(LoadableModel):
    @store_config_args
    def __init__(self,
                 n_classes: int,
                 head_latent_space=64,
                 channels=(64, 128, 256, 512),
                 strides=(2, 2, 2),
                 num_res_units=2,
                 norm='instance',
                 act='leakyrelu',
                 uv_channels_per_class=2):
        super(UVUNet, self).__init__()
        self.uv_channels = uv_channels_per_class
        self.unet = nets.UNet(
            spatial_dims=2,
            in_channels=1,
            out_channels=head_latent_space,
            channels=channels,
            strides=strides,
            num_res_units=num_res_units,
            norm=(norm, {'affine': True}),
            act=act,
            bias=False
        )

        self.seg_head = nn.Sequential(
            cat_residual_units(num_res_units, channels[0], head_latent_space, norm, act),
            nn.Conv2d(head_latent_space, n_classes, kernel_size=1, bias=True)
        )

        self.uv_head = nn.Sequential(
            cat_residual_units(num_res_units, channels[0], head_latent_space, norm, act),
            nn.Conv2d(head_latent_space, n_classes * self.uv_channels, kernel_size=1, bias=True),
            nn.Tanh()
        )

    def forward(self, x):
        features = self.unet(x)
        seg = self.seg_head(features)
        uv = self.uv_head(features)
        uv = uv.view(uv.shape[0], -1, self.uv_channels, uv.shape[2], uv.shape[3])  # for each class predict UV (B, C, UV, H, W)
        return seg, uv  # (B, C, H, W), (B, C, UV, H, W)

    @torch.no_grad()
    def predict(self, x, mask_uv=True):
        seg, uv = self.forward(x)
        seg = seg.sigmoid() > 0.5

        # mask uv values to valid segmentation area
        if mask_uv:
            mask = seg.unsqueeze(2).expand_as(uv)
            uv = torch.where(mask, uv, torch.nan)

        return seg, uv


if __name__ == '__main__':
    from torchinfo import summary
    import torch

    model = UVUNet(n_classes=17)
    print(model)
    summary(model, (1, 1, 256, 256))
    # seg, uv = model(torch.randn(1, 1, 256, 256))
    # print(seg.shape, uv.shape)
    # seg_hat, uv_hat = model.predict(torch.randn(1, 1, 256, 256))

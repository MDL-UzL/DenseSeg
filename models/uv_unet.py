from monai.networks import nets, blocks
from models.modelio import LoadableModel, store_config_args
from torch import nn
import torch


class UVUNet(LoadableModel):
    @store_config_args
    def __init__(self,
                 n_classes: int,
                 head_latent_space=32,
                 channels=(32, 64, 128, 256, 512),
                 strides=(2, 2, 2, 2),
                 num_res_units=2,
                 norm='instance',
                 act='leakyrelu'):
        super(UVUNet, self).__init__()
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
            blocks.ResidualUnit(
                spatial_dims=2,
                in_channels=head_latent_space,
                out_channels=head_latent_space * 2,
                kernel_size=3,
                norm=(norm, {'affine': True}),
                act=act,
                bias=False
            ),
            nn.Conv2d(head_latent_space * 2, n_classes, kernel_size=1, bias=True)
        )

        self.uv_head = nn.Sequential(
            blocks.ResidualUnit(
                spatial_dims=2,
                in_channels=head_latent_space,
                out_channels=head_latent_space * 2,
                kernel_size=3,
                norm=(norm, {'affine': True}),
                act=act,
                bias=False
            ),
            nn.Conv2d(head_latent_space * 2, n_classes * 2, kernel_size=1, bias=True),
            nn.Tanh()
        )

    def forward(self, x):
        features = self.unet(x)
        seg = self.seg_head(features)
        uv = self.uv_head(features)
        uv = uv.view(uv.shape[0], -1, 2, uv.shape[2], uv.shape[3])  # for each class predict UV (B, C, 2, H, W)
        return seg, uv # (B, C, H, W), (B, C, 2, H, W)

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
    seg, uv = model(torch.randn(1, 1, 256, 256))
    print(seg.shape, uv.shape)
    seg_hat, uv_hat = model.predict(torch.randn(1, 1, 256, 256))

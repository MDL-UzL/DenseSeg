from monai.networks import nets
from torch import nn

from models.modelio import LoadableModel, store_config_args
from models.uv_unet import cat_residual_units


class KeypointUNet(LoadableModel):
    @store_config_args
    def __init__(self,
                 n_kpts: int,
                 channels=(64, 128, 256, 512),
                 strides=(2, 2, 2),
                 num_res_units=2,
                 norm='instance',
                 act='leakyrelu'):
        super(KeypointUNet, self).__init__()
        self.unet = nets.UNet(
            spatial_dims=2,
            in_channels=1,
            out_channels=channels[0],
            channels=channels,
            strides=strides,
            num_res_units=num_res_units,
            norm=(norm, {'affine': True}),
            act=act,
            bias=False
        )

        self.heatmap_regressor = nn.Conv2d(self.unet.out_channels, n_kpts, kernel_size=1, bias=True)

    def forward(self, x):
        features = self.unet(x)
        y = self.heatmap_regressor(features)
        return y


class KeypointSegUNet(LoadableModel):
    @store_config_args
    def __init__(self,
                 n_kpts: int,
                 n_classes: int,
                 head_latent_space=64,
                 channels=(64, 128, 256, 512),
                 strides=(2, 2, 2),
                 num_res_units=2,
                 norm='instance',
                 act='leakyrelu'):
        super(KeypointSegUNet, self).__init__()
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

        self.heatmap_regressor = nn.Sequential(
            cat_residual_units(num_res_units, channels[0], head_latent_space, norm, act),
            nn.Conv2d(head_latent_space, n_kpts, kernel_size=1, bias=True)
        )

    def forward(self, x):
        features = self.unet(x)
        seg = self.seg_head(features)
        kpts = self.heatmap_regressor(features)
        return seg, kpts


if __name__ == '__main__':
    import torch
    from torchinfo import summary
    from evaluation.clearml_ids import grazer_model_ids
    from clearml import InputModel

    #model = KeypointUNet.load(InputModel(grazer_model_ids['heatmap']).get_weights(), 'cpu')
    model = KeypointSegUNet(166, 4)
    print(model)
    seg_hat, kpts_hat = model(torch.randn(1, 1, 256, 256))
    print(seg_hat.shape, kpts_hat.shape)
    summary(model, (1, 1, 256, 256))

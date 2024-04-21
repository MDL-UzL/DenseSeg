from monai.networks import nets
from torch import nn

from models.modelio import LoadableModel, store_config_args


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


if __name__ == '__main__':
    from torchinfo import summary

    model = KeypointUNet(n_kpts=160)
    print(model)
    summary(model, (1, 1, 256, 256))

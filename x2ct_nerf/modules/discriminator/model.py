from x2ct_nerf.modules.discriminator.base import *

class ProgressiveDiscriminator(nn.Module):
    """Implement of a progressive growing discriminator with ResidualCoordConv Blocks"""

    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList(
        [
            ResidualCoordConvBlock(16, 32, downsample=True),   # 512x512 -> 256x256
            ResidualCoordConvBlock(32, 64, downsample=True),   # 256x256 -> 128x128
            ResidualCoordConvBlock(64, 128, downsample=True),  # 128x128 -> 64x64
            ResidualCoordConvBlock(128, 256, downsample=True), # 64x64   -> 32x32
            ResidualCoordConvBlock(256, 400, downsample=True), # 32x32   -> 16x16
            ResidualCoordConvBlock(400, 400, downsample=True), # 16x16   -> 8x8
            ResidualCoordConvBlock(400, 400, downsample=True), # 8x8     -> 4x4
            ResidualCoordConvBlock(400, 400, downsample=True), # 4x4     -> 2x2
        ])

        self.fromRGB = nn.ModuleList(
        [
            AdapterBlock(16),
            AdapterBlock(32),
            AdapterBlock(64),
            AdapterBlock(128),
            AdapterBlock(256),
            AdapterBlock(400),
            AdapterBlock(400),
            AdapterBlock(400),
            AdapterBlock(400)
        ])
        self.final_layer = nn.Conv2d(400, 1, 2)
        self.img_size_to_layer = {2:8, 4:7, 8:6, 16:5, 32:4, 64:3, 128:2, 256:1, 512:0}

    def forward(self, input, alpha):
        if input.shape[1] == 1:
            input = input.repeat(1, 3, 1, 1)
        start = self.img_size_to_layer[input.shape[-1]]

        x = self.fromRGB[start](input)
        for i, layer in enumerate(self.layers[start:]):
            if i == 1:
                x = alpha * x + (1 - alpha) * self.fromRGB[start+1](F.interpolate(input, scale_factor=0.5, mode='nearest'))
            x = layer(x)

        output = self.final_layer(x).reshape(x.shape[0], 1)
        return output
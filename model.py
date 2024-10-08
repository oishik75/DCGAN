import torch
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, channels_img, features_d) -> None:
        super().__init__()

        # Input: N x channels_img x 64 x 64
        self.disc = nn.Sequential(
            self._block(channels_img, features_d, kernel_size=4, stride=2, padding=1, batch_norm=False), #32x32
            self._block(features_d, features_d*2, kernel_size=4, stride=2, padding=1), # 16x16
            self._block(features_d*2, features_d*4, kernel_size=4, stride=2, padding=1), # 8x8
            self._block(features_d*4, features_d*8, kernel_size=4, stride=2, padding=1), # 4x4
            nn.Conv2d(features_d*8, 1, kernel_size=4, stride=2, padding=0), # 1x1
            nn.Sigmoid()
        )

    def _block(self, in_channels, out_channels, kernel_size, stride, padding, batch_norm=True):
        layers = []
        bias = False if batch_norm else True
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias))
        if batch_norm:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.LeakyReLU(0.2))
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.disc(x)
    
class Generator(nn.Module):
    def __init__(self, z_dim, channels_img, features_g) -> None:
        super().__init__()

        # Input: N x z_dim x 1 x 1
        self.gen = nn.Sequential(
            self._block(z_dim, features_g*16, 4, 1, 0), # Nxf_g*16x4x4
            self._block(features_g*16, features_g*8, 4, 2, 1), # 8x8
            self._block(features_g*8, features_g*4, 4, 2, 1), # 16x16
            self._block(features_g*4, features_g*2, 4, 2, 1), # 32x32
            self._block(features_g*2, channels_img, 4, 2, 1, batch_norm=False), # 64x64
            nn.Tanh()
        )

    def _block(self, in_channels, out_channels, kernel_size, stride, padding, batch_norm=True):
        layers = []
        bias = False if batch_norm else True
        layers.append(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        )
        if batch_norm:
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU())
        return nn.Sequential(*layers)
    
    def forward(self, x):
        return self.gen(x)
    
def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)

def test():
    N, in_channels, H, W = 8, 3, 64, 64
    # Test Discriminator
    x = torch.randn((N, in_channels, H, W))
    disc = Discriminator(in_channels, 8)
    initialize_weights(disc)
    assert disc(x).shape == (N, 1, 1, 1), f"Incorrect Discriminator shape output. Expected: {(N, 1, 1, 1)}. Actual: {disc(x).shape}"
    # Test Generator
    z_dim = 100
    z = torch.randn((N, z_dim, 1, 1))
    gen = Generator(z_dim, in_channels, 8)
    initialize_weights(gen)
    assert gen(z).shape == (N, in_channels, H, W), f"Incorrect Generator shape output. Expected: {(N, in_channels, H, W)}. Actual: {gen(x).shape}"

if __name__ == "__main__":
    test()
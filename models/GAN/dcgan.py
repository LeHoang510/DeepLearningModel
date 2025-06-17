import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, latent_dim):
        super(Generator, self).__init__()

        self.layers = nn.Sequential(
            # Input: latent_dim x 1 x 1
            nn.Conv2dTranspose(latent_dim, 1024, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(1024),
            nn.ReLU(True),
            # Output: 1024 x 4 x 4
            nn.Conv2dTranspose(1024, 512, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            # Output: 512 x 8 x 8
            nn.Conv2dTranspose(512, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            # Output: 256 x 16 x 16
            nn.Conv2dTranspose(256, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            # Output: 128 x 32 x 32
            nn.Conv2dTranspose(128, 3, kernel_size=4, stride=2, padding=1, bias=False),
            # Output: 3 x 64 x 64
            nn.Tanh()
        )

    def forward(self, x):
        # x: (batch_size, latent_dim, 1, 1)
        out = self.layers(x)
        return out

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2dTranspose) or isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.layers = nn.Sequential(
            # Input: 3 x 64 x 64
            nn.Conv2d(3, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # Output: 128 x 32 x 32
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            # Output: 256 x 16 x 16
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            # Output: 512 x 8 x 8
            nn.Conv2d(512, 1024, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.2, inplace=True),
            # Output: 1024 x 4 x 4
            nn.Conv2d(1024, 1, kernel_size=4, stride=1, padding=0, bias=False),
            # Output: 1 x 1 x 1
            nn.Sigmoid()
        )

    def forward(self, x):
        # x: (batch_size, 3, 64, 64)
        out = self.layers(x)
        return out.view(-1, 1)  # Flatten to (batch_size, 1)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

import torch
import torch.nn as nn
import torch.nn.functional as F

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(100, 256),
            nn.ReLU(inplace=True),

            nn.Linear(256, 512),
            nn.ReLU(inplace=True),

            nn.Linear(512, 1024),
            nn.ReLU(inplace=True),

            nn.Linear(1024, 784),
            nn.Tanh()
        )

        self._init_weights()

    def forward(self, x):
        out = self.layers(x)
        return out

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(784, 512),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(256, 128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        self._init_weights()

    def forward(self, x):
        out = self.layers(x)
        return out

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

# print(Discriminator([784, 512, 256, 128, 1]))

# Generator(
#   (layers): ModuleList(
#     (0): Linear(in_features=100, out_features=256, bias=True)
#     (1): ReLU(inplace=True)
#     (2): Linear(in_features=256, out_features=512, bias=True)
#     (3): ReLU(inplace=True)
#     (4): Linear(in_features=512, out_features=1024, bias=True)
#     (5): ReLU(inplace=True)
#     (6): Linear(in_features=1024, out_features=784, bias=True)
#     (7): Tanh()
#   )
# )

# Discriminator(
#   (layers): ModuleList(
#     (0): Linear(in_features=784, out_features=512, bias=True)
#     (1): LeakyReLU(negative_slope=0.2, inplace=True)
#     (2): Linear(in_features=512, out_features=256, bias=True)
#     (3): LeakyReLU(negative_slope=0.2, inplace=True)
#     (4): Linear(in_features=256, out_features=128, bias=True)
#     (5): LeakyReLU(negative_slope=0.2, inplace=True)
#     (6): Linear(in_features=128, out_features=1, bias=True)
#     (7): Sigmoid()
#   )
# )

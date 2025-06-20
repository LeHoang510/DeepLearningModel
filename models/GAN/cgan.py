"""
Conditional Generative Adversarial Network (CGAN) implementation.
"""
import torch
import torch.nn as nn


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.embedding = nn.Embedding(10, 10)

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

    def forward(self, x, label):
        pass

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.embedding = nn.Embedding(10, 10)
        self.layers = nn.Sequential(
            
        )

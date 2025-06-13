"""
VGG16 architecture implementation using PyTorch.
Summary:
    - 13 convolutional layers with 3x3 kernels, followed by 3 fully connected layers.
    - Uses ReLU activations and Dropout.
Input size: 3x224x224
Layers:
    - features:
        - Conv Block 1: 2 x (64 filters, 3x3 kernel, stride 1, padding 1 + ReLU) + MaxPooling
        - Conv Block 2: 2 x (128 filters, 3x3 kernel, stride 1, padding 1 + ReLU) + MaxPooling
        - Conv Block 3: 3 x (256 filters, 3x3 kernel, stride 1, padding 1 + ReLU) + MaxPooling
        - Conv Block 4: 3 x (512 filters, 3x3 kernel, stride 1, padding 1 + ReLU) + MaxPooling
        - Conv Block 5: 3 x (512 filters, 3x3 kernel, stride 1, padding 1 + ReLU) + MaxPooling
    - classifier:
        - FC1: 512*7*7 -> 4096 + ReLU + Dropout
        - FC2: 4096 -> 4096 + ReLU + Dropout
        - FC3: 4096 -> num_classes
"""

import torch
import torch.nn as nn

class VGG16(nn.Module):
    def __init__(self, num_classes=1000):
        super(VGG16, self).__init__()
        self.features = self._make_features()
        self.classifier = self._make_classifier(num_classes)
    
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)  
        x = self.classifier(x)
        return x
    
    def _make_features(self):
        conv_block1 = self._make_block(2, 3, 64)
        conv_block2 = self._make_block(2, 64, 128)
        conv_block3 = self._make_block(3, 128, 256)
        conv_block4 = self._make_block(3, 256, 512)
        conv_block5 = self._make_block(3, 512, 512)
        return nn.Sequential(
            conv_block1,
            conv_block2,
            conv_block3,
            conv_block4,
            conv_block5
        )

    def _make_classifier(self, num_classes):
        fc1 = nn.Sequential(
            nn.Dropout(),
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
        )
        fc2 = nn.Sequential(
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
        )
        fc3 = nn.Linear(4096, num_classes)
        return nn.Sequential(
            fc1,
            fc2,
            fc3
        )

    def _make_block(self, num_convs, in_channels, out_channels):
        layers = []
        for _ in range(num_convs):
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1))
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU(inplace=True))
            in_channels = out_channels
        layers.append(nn.MaxPool2d(kernel_size=3, stride=1))
        return nn.Sequential(*layers)
    

# vgg16 = VGG16(num_classes=1000)
# print(vgg16)  
"""
AlexNet model implementation using PyTorch.
Summary:
    - 5 convolutional layers (11, 5, 3) followed by 3 fully connected layers.
    - Uses ReLU activations, Batch Normalization, and Dropout.
Input size: 3x224x224
Layers:
    - features:
        - Conv1: 96 filters, 11x11 kernel, stride 4, padding 0 + ReLU + BN + MaxPool
        - Conv2: 256 filters, 5x5 kernel, stride 1, padding 2 + ReLU + BN + MaxPool
        - Conv3: 384 filters, 3x3 kernel, stride 1, padding 1 + ReLU
        - Conv4: 384 filters, 3x3 kernel, stride 1, padding 1 + ReLU
        - Conv5: 256 filters, 3x3 kernel, stride 1, padding 1 + ReLU + BN + MaxPool
    - classifier:
        - FC1: Dropout + 256*6*6 -> 4096 + ReLU
        - FC2: Dropout + 4096 -> 4096 + ReLU
        - FC3: 4096 -> num_classes
"""

import torch
import torch.nn as nn

class AlexNet(nn.Module):
    def __init__(self, num_classes=1000):
        super(AlexNet, self).__init__()
        self.features = self._make_features()
        self.classifier = self._make_classifier(num_classes)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)  # Flatten the tensor while keeping the batch dimension
        x = self.classifier(x)
        return x

    def _make_features(self):
        layer1 = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=0),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        layer2 = nn.Sequential(
            nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        layer3 = nn.Sequential(
            nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True)
        )
        layer4 = nn.Sequential(
            nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True)
        )
        layer5 = nn.Sequential(
            nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=1)
        )
        return nn.Sequential(layer1, layer2, layer3, layer4, layer5)

    def _make_classifier(self, num_classes):
        fc1 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256*6*6, 4096),
            nn.ReLU(inplace=True)
        )
        fc2 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True)
        )
        fc3 = nn.Sequential(
            nn.Linear(4096, num_classes)
        )
        return nn.Sequential(fc1, fc2, fc3)


# alexnet = AlexNet(num_classes=1000)
# print(alexnet)

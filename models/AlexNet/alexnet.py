"""
AlexNet archtecture implementation.
Input size: 3x224x224
Subsampling using Max Pooling and ReLU activation.
Layers:
- layer1: Conv2D(3, 96, kernel_size=11, stride=4, padding=0), batch normalization, ReLU, max pooling
- layer2: Conv2D(96, 256, kernel_size=5, stride=1, padding=2), batch normalization, ReLU, max pooling
- layer3: Conv2D(256, 384, kernel_size=3, stride=1, padding=1), batch normalization, ReLU
- layer4: Conv2D(384, 384, kernel_size=3, stride=1, padding=1), batch normalization, ReLU
- layer5: Conv2D(384, 256, kernel_size=3, stride=1, padding=1), batch normalization, ReLU, max pooling
- layer6: Flatten -> Linear(256*6*6, 4096), ReLU, dropout
- layer7: Linear(4096, 4096), ReLU, dropout
- layer8: Linear(4096, num_classes)

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
        x = x.view(x.size(0), -1)
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


# model = AlexNet(num_classes=1000)
# print(model)

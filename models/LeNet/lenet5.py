"""
LeNet-5 model implementation using PyTorch.
Summary:
    - 2 convolutional layers with 5x5 kernel, followed by 2 fully connected layers.
Input size: 1x32x32
Layers:
    - features:
        - Conv1: 6 filters, 5x5 kernel, stride 1, padding 0 + ReLU + MaxPool
        - Conv2: 16 filters, 5x5 kernel, stride 1, padding 0 + ReLU + MaxPool
    - classifier:
        - FC1: 16*5*5 -> 120 + ReLU
        - FC2: 120 -> 84 + ReLU
        - FC3: 84 -> num_classes
"""
import torch
import torch.nn as nn

class LeNet5(nn.Module):
    def __init__(self, num_classes=10):
        super(LeNet5, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(6),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(16 * 5 * 5, 120),
            nn.ReLU(inplace=True),
            nn.Linear(120, 84),
            nn.ReLU(inplace=True),
            nn.Linear(84, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
    
# lenet = LeNet5(num_classes=10)
# print(lenet)
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
        self.features = self._make_features()
        self.classifier = self._make_classifier(num_classes)
        self._init_weights()

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def _make_features(self):
        conv1 = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(6),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        conv2 = nn.Sequential(
            nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        return nn.Sequential(conv1, conv2)

    def _make_classifier(self, num_classes):
        fc1 = nn.Sequential(
            nn.Linear(16 * 5 * 5, 120),
            nn.ReLU(inplace=True),
            nn.Dropout()
        )
        fc2 = nn.Sequential(
            nn.Linear(120, 84),
            nn.ReLU(inplace=True),
            nn.Dropout()
        )
        fc3 = nn.Linear(84, num_classes)
        return nn.Sequential(fc1, fc2, fc3)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


# lenet = LeNet5(num_classes=10)
# print(lenet)
# LeNet5(
#   (features): Sequential(
#     (0): Sequential(
#       (0): Conv2d(1, 6, kernel_size=(5, 5), stride=(1, 1))
#       (1): BatchNorm2d(6, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (2): ReLU(inplace=True)
#       (3): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
#     )
#     (1): Sequential(
#       (0): Conv2d(6, 16, kernel_size=(5, 5), stride=(1, 1))
#       (1): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (2): ReLU(inplace=True)
#       (3): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
#     )
#   )
#   (classifier): Sequential(
#     (0): Sequential(
#       (0): Linear(in_features=400, out_features=120, bias=True)
#       (1): ReLU(inplace=True)
#       (2): Dropout(p=0.5, inplace=False)
#     )
#     (1): Sequential(
#       (0): Linear(in_features=120, out_features=84, bias=True)
#       (1): ReLU(inplace=True)
#       (2): Dropout(p=0.5, inplace=False)
#     )
#     (2): Linear(in_features=84, out_features=10, bias=True)
#   )
# )

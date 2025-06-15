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
        self._init_weights()

    def forward(self, x):
        out = self.features(x)
        out = torch.flatten(out, 1)  # Flatten the tensor while keeping the batch dimension
        out = self.classifier(out)
        return out

    def _make_features(self):
        conv1 = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=0, bias=False),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        conv2 = nn.Sequential(
            nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=2, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        conv3 = nn.Sequential(
            nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True)
        )
        conv4 = nn.Sequential(
            nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True)
        )
        conv5 = nn.Sequential(
            nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=1)
        )
        return nn.Sequential(conv1, conv2, conv3, conv4, conv5)

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
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

# alexnet = AlexNet(num_classes=1000)
# print(alexnet)
# AlexNet(
#   (features): Sequential(
#     (0): Sequential(
#       (0): Conv2d(3, 96, kernel_size=(11, 11), stride=(4, 4))
#       (1): BatchNorm2d(96, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (2): ReLU(inplace=True)
#       (3): MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)
#     )
#     (1): Sequential(
#       (0): Conv2d(96, 256, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
#       (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (2): ReLU(inplace=True)
#       (3): MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)
#     )
#     (2): Sequential(
#       (0): Conv2d(256, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#       (1): BatchNorm2d(384, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (2): ReLU(inplace=True)
#     )
#     (3): Sequential(
#       (0): Conv2d(384, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#       (1): BatchNorm2d(384, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (2): ReLU(inplace=True)
#     )
#     (4): Sequential(
#       (0): Conv2d(384, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#       (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (2): ReLU(inplace=True)
#       (3): MaxPool2d(kernel_size=3, stride=1, padding=0, dilation=1, ceil_mode=False)
#     )
#   )
#   (classifier): Sequential(
#     (0): Sequential(
#       (0): Dropout(p=0.5, inplace=False)
#       (1): Linear(in_features=9216, out_features=4096, bias=True)
#       (2): ReLU(inplace=True)
#     )
#     (1): Sequential(
#       (0): Dropout(p=0.5, inplace=False)
#       (1): Linear(in_features=4096, out_features=4096, bias=True)
#       (2): ReLU(inplace=True)
#     )
#     (2): Sequential(
#       (0): Linear(in_features=4096, out_features=1000, bias=True)
#     )
#   )
# )

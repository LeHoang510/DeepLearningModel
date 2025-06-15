"""
GoogleNet model implementation using PyTorch.
Summary:
    - 3 main convolutional layers followed by 9 Inception blocks.
    - 2 Auxiliary classifiers for intermediate supervision.
Input size: 3x224x224
Layers:
    - conv1: 64 filters, 7x7 kernel, stride 2, padding 3 + ReLU + MaxPool
    - conv2: 64 filters, 1x1 kernel + ReLU
    - conv3: 192 filters, 3x3 kernel, padding 1 + ReLU + MaxPool
    - inception blocks: multiple branches with different kernel sizes and pooling
    - auxiliary classifiers: additional supervision for intermediate layers
    - avgpool: AdaptiveAvgPool2d to (1, 1)
    - dropout: Dropout with p=0.2
    - fc: Fully connected layer to num_classes

"""

import torch
import torch.nn as nn

class GoogLeNet(nn.Module):
    def __init__(self, num_classes=1000):
        super(GoogLeNet, self).__init__()

        self.conv1 = BasicBlock(3, 64, kernel_size=7, stride=2, padding=3)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv2 = BasicBlock(64, 64, kernel_size=1)
        self.conv3 = BasicBlock(64, 192, kernel_size=3, padding=1)
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.inception1 = InceptionBlock(192, 64, 96, 128, 16, 32, 32)
        self.inception2 = InceptionBlock(256, 128, 128, 192, 32, 96, 64)
        self.maxpool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.inception3 = InceptionBlock(480, 192, 96, 208, 16, 48, 64)
        self.auxiliary1 = AuxiliaryClassifier(512, num_classes)
        self.inception4 = InceptionBlock(512, 160, 112, 224, 24, 64, 64)

        self.inception5 = InceptionBlock(512, 128, 128, 256, 24, 64, 64)
        self.inception6 = InceptionBlock(512, 112, 144, 288, 32, 64, 64)
        self.auxiliary2 = AuxiliaryClassifier(528, num_classes)
        self.inception7 = InceptionBlock(528, 256, 160, 320, 32, 128, 128)
        self.maxpool4 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.inception8 = InceptionBlock(832, 256, 160, 320, 32, 128, 128)
        self.inception9 = InceptionBlock(832, 384, 192, 384, 48, 128, 128)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(1024, num_classes)

        self._init_weights()

    def forward(self, x):
        out = self.conv1(x)
        out = self.maxpool1(out)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.maxpool2(out)

        out = self.inception1(out)
        out = self.inception2(out)
        out = self.maxpool3(out)

        out = self.inception3(out)
        aux1 = self.auxiliary1(out)
        out = self.inception4(out)
        out = self.inception5(out)
        out = self.inception6(out)

        aux2 = self.auxiliary2(out)
        out = self.inception7(out)
        out = self.maxpool4(out)
        out = self.inception8(out)
        out = self.inception9(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.dropout(out)
        out = self.fc(out)

        return out, aux2, aux1

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


class InceptionBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out1x1,
        out3x3_reduce,
        out3x3,
        out5x5_reduce,
        out5x5,
        pool_proj
    ):
        super(InceptionBlock, self).__init__()
        self.branch1 = BasicBlock(in_channels, out1x1, kernel_size=1)
        self.branch2 = nn.Sequential(
            BasicBlock(in_channels, out3x3_reduce, kernel_size=1),
            BasicBlock(out3x3_reduce, out3x3, kernel_size=3, padding=1)
        )
        self.branch3 = nn.Sequential(
            BasicBlock(in_channels, out5x5_reduce, kernel_size=1),
            BasicBlock(out5x5_reduce, out5x5, kernel_size=5, padding=2)
        )
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            BasicBlock(in_channels, pool_proj, kernel_size=1)
        )

    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)

        outputs = torch.cat([branch1, branch2, branch3, branch4], 1)
        return outputs

class AuxiliaryClassifier(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(AuxiliaryClassifier, self).__init__()

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((4, 4)),
            BasicBlock(in_channels, 128, kernel_size=1),
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.7),
            nn.Linear(1024, num_classes)
        )

    def forward(self, x):
        return self.classifier(x)

class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, **kwargs, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)

# googlenet = GoogLeNet(num_classes=1000)
# print(googlenet)

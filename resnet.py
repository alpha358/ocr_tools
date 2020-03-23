# ==============================================================================
# ============================= CREDIT: JULIUS R. ==============================
# ==============================================================================
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms

def init_linear(m):
    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    if m.bias is not None: nn.init.zeros_(m.bias)

class ConvBlock(nn.Sequential):
    def __init__(self, filters1, filters2, kernel_size, stride=1, include_relu=True, zero_bn=False):
        self.zero_bn = zero_bn
        padding = (kernel_size - 1) // 2
        layers = [nn.Conv2d(filters1, filters2, kernel_size, stride=stride, padding=padding, bias=False),
                  nn.BatchNorm2d(filters2)]
        if include_relu:
            layers.append(nn.ReLU(inplace=True))

        super().__init__(*layers)

    def reset_parameters(self):
        init_linear(self[0])
        self[1].reset_parameters()
        if self.zero_bn:
            nn.init.zeros_(self[1].weight)


class ResnetBlock(nn.Module):
    def __init__(self, prev_filters, filters, stride=1, bottleneck=True):
        super().__init__()

        if bottleneck:
            self.residual = self.bottleneck_residual(prev_filters, filters, stride)
        else:
            self.residual = self.basic_residual(prev_filters, filters, stride)

        if stride == 1:
            self.shortcut = lambda x: x
        else:
            self.shortcut = nn.Sequential(
                nn.AvgPool2d(stride),
                ConvBlock(prev_filters, filters, 1, include_relu=False)
            )

    def basic_residual(self, prev_filters, filters, stride):
        return nn.Sequential(
            ConvBlock(prev_filters, filters, 3, stride=stride),
            ConvBlock(filters, filters, 3, include_relu=False, zero_bn=True)
        )

    def bottleneck_residual(self, prev_filters, filters, stride):
        bottleneck_filters = filters // 4
        return nn.Sequential(
            ConvBlock(prev_filters, bottleneck_filters, 1),
            ConvBlock(bottleneck_filters, bottleneck_filters, 3, stride=stride),
            ConvBlock(bottleneck_filters, filters, 1, include_relu=False, zero_bn=True)
        )

    def forward(self, x):
        out = self.residual(x) + self.shortcut(x)
        out = F.relu(out, inplace=True)
        return out


class ResNet(nn.Sequential):
    def __init__(self, repetitions, bottleneck=True, head=False, classes=2):
        super().__init__()

        layers = [ConvBlock(3, 32, 3, stride=2),
                  ConvBlock(32, 32, 3),
                  ConvBlock(32, 64, 3)]

        prev_filters = 64
        init_filters = 256 if bottleneck else 64
        for stage, rep in enumerate(repetitions):
            filters = init_filters * (2**stage)
            stride = 2
            for _ in range(rep):
                layers.append(ResnetBlock(prev_filters, filters, stride, bottleneck))
                prev_filters = filters
                stride = 1

        if head:
            layers += [nn.AdaptiveAvgPool2d(1),
                       nn.Flatten(),
                       nn.Linear(prev_filters, classes)]

        super().__init__(*layers)
        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, ConvBlock) or isinstance(m, nn.Linear):
                m.reset_parameters()

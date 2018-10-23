import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from models import base

np.random.seed(2 ** 10)

class FlatResNet(nn.Module):

    def seed(self, x):
        # x = self.relu(self.bn1(self.conv1(x))) -- CIFAR
        # x = self.maxpool(self.relu(self.bn1(self.conv1(x)))) -- ImageNet
        raise NotImplementedError

    def forward(self, x, stochastic=False):
        if not stochastic:
            x = self.seed(x)

            for segment, num_blocks in enumerate(self.layer_config):
                for b in range(num_blocks):
                    residual = self.ds[segment](x) if b == 0 else x
                    x = F.relu(residual + self.blocks[segment][b](x))

            x = self.avgpool(x)
            x = x.view(x.size(0), -1)
            x = self.fc(x)
            return x
        else:
            x = self.seed(x)
            step = 0.5 / sum(self.layer_config)
            p = 1.0
            for segment, num_blocks in enumerate(self.layer_config):
                for b in range(num_blocks):
                    action = np.random.choice([1, 0], p=[p, 1-p])
                    p = p - step
                    residual = self.ds[segment](x) if b==0 else x
                    if action == 0:
                        x = residual
                    else:
                        x = F.relu(residual + self.blocks[segment][b](x))

            x = self.avgpool(x)
            x = x.view(x.size(0), -1)
            x = self.fc(x)
            return x


class FlatResNet32(FlatResNet):

    def __init__(self, block, layers, num_classes=10):
        super().__init__()

        self.inplanes = 16
        self.conv1 = base.conv3x3(3, 16)
        self.bn1 = nn.BatchNorm2d(16, track_running_stats=False)
        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(8)

        strides = [1, 2, 2]
        filt_sizes = [16, 32, 64]
        self.blocks, self.ds = [], []
        for idx, (filt_size, num_blocks, stride) in enumerate(zip(filt_sizes, layers, strides)):
            blocks, ds = self._make_layer(block, filt_size, num_blocks, stride=stride)
            self.blocks.append(nn.ModuleList(blocks))
            self.ds.append(ds)

        self.blocks = nn.ModuleList(self.blocks)
        self.ds = nn.ModuleList(self.ds)
        self.fc = nn.Linear(64 * block.expansion, num_classes)
        self.fc_dim = 64 * block.expansion

        self.layer_config = layers

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def seed(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        return x

    def _make_layer(self, block, planes, blocks, stride=1):

        downsample = nn.Sequential()
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = base.DownsampleB(self.inplanes, planes * block.expansion, stride)

        layers = [block(self.inplanes, planes, stride)]
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, 1))

        return layers, downsample

# TODO: FlatResNet224 for ImageNet

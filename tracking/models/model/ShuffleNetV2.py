import math
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models._utils as _utils
from torch.autograd import Variable
from torch.nn import init


def conv_bn(inp, oup, kernel_size , stride , pad):
    return nn.Sequential(
        nn.Conv2d(inp, oup, kernel_size, stride, pad, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True)
    )


def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True)
    )


def channel_shuffle(x, groups=2):
    batchsize, num_channels, height, width = x.data.size()
    channels_per_group = num_channels // groups
    x = x.view(batchsize, groups, channels_per_group, height, width)
    x = torch.transpose(x, 1, 2).contiguous()
    x = x.view(batchsize, -1, height, width)
    return x


class ShuffleV2Block(nn.Module):
    def __init__(self, inp, oup, mid_channels, *, ksize, stride):
        super(ShuffleV2Block, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        self.mid_channels = mid_channels
        self.ksize = ksize
        pad = ksize // 2
        self.pad = pad
        self.inp = inp

        outputs = oup - inp

        branch_main = [
            # pw
            nn.Conv2d(inp, mid_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            # dw
            nn.Conv2d(mid_channels, mid_channels, ksize, stride, pad, groups=mid_channels, bias=False),
            nn.BatchNorm2d(mid_channels),
            # pw-linear
            nn.Conv2d(mid_channels, outputs, 1, 1, 0, bias=False),
            nn.BatchNorm2d(outputs),
            nn.ReLU(inplace=True),
        ]
        self.branch_main = nn.Sequential(*branch_main)

        if stride == 2:
            branch_proj = [
                # dw
                nn.Conv2d(inp, inp, ksize, stride, pad, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                # pw-linear
                nn.Conv2d(inp, inp, 1, 1, 0, bias=False),
                nn.BatchNorm2d(inp),
                nn.ReLU(inplace=True),
            ]
            self.branch_proj = nn.Sequential(*branch_proj)
        else:
            self.branch_proj = None

    def forward(self, old_x):
        if self.stride==1:
            x_proj, x = self.channel_shuffle(old_x)
            return torch.cat((x_proj, self.branch_main(x)), 1)
        elif self.stride==2:
            x_proj = old_x
            x = old_x
            return torch.cat((self.branch_proj(x_proj), self.branch_main(x)), 1)

    def channel_shuffle(self, x, g = 2):
        x = x.reshape(x.shape[0], g, x.shape[1] // g, x.shape[2], x.shape[3])
        x = x.permute(0, 2, 1, 3, 4)
        x = x.reshape(x.shape[0], -1, x.shape[3], x.shape[4])
        x_proj = x[:, :(x.shape[1] // 2), :, :]
        x = x[:, (x.shape[1] // 2):, :, :]
        return x_proj, x


class ShuffleNetV2(nn.Module):
    def __init__(self, cfg, **kwargs):
        super(ShuffleNetV2, self).__init__()
        input_size = cfg['image_size'][0]
        self.stage_repeats = cfg['stage_repeats']
        width_mult = cfg['width_mult']
        n_class = cfg['n_class']
        assert input_size % 32 == 0

        num_layers = [4, 8, 4]
        self.num_layers = num_layers
        channels = [24, 132, 264, 528, 1056]
        self.channels = channels

        # building first layer
        self.conv1 = conv_bn(
            3, channels[0], kernel_size=3, stride=2,pad = 1
        )
        self.maxpool = nn.MaxPool2d(
            kernel_size=3, stride=2, padding=1,
        )


        self.layer1 = self._make_layer(
            num_layers[0], channels[0], channels[1], **kwargs)
        self.layer2 = self._make_layer(
            num_layers[1], channels[1], channels[2], **kwargs)
        self.layer3 = self._make_layer(
            num_layers[2], channels[2], channels[3], **kwargs)
        # if len(self.channels) == 5:
        #     self.conv5 = conv_bn(
        #         channels[3], channels[4], kernel_size=1, stride=1 ,pad=0 )

        # building last several layers
        # self.conv_last = conv_1x1_bn(input_channel, self.channels[-1])
        self.globalpool = nn.Sequential(nn.AvgPool2d(int(input_size/32)))
        self.classifier = nn.Sequential(nn.Linear(self.channels[-1], n_class))


    def _make_layer(self, num_layers, in_channels, out_channels, **kwargs):
        layers = []
        for i in range(num_layers):
            if i == 0:
                layers.append(ShuffleV2Block(in_channels, out_channels, mid_channels=out_channels // 2, ksize=5, stride=2))
            else:
                layers.append(ShuffleV2Block(in_channels // 2, out_channels,
                                                    mid_channels=out_channels // 2, ksize=5, stride=1))
            in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)

        c3 = self.layer1(x)
        c4 = self.layer2(c3)
        c5 = self.layer3(c4)
        # if len(self.channels) == 5:
        #     c5 = self.conv5(c5)
        # x = self.conv_last(x)
        x = self.globalpool(c5)
        x = x.view(-1, self.channels[-1])
        x = self.classifier(x)
        return x


def shufflenetv2(width_mult=1.):
    model = ShuffleNetV2(width_mult=width_mult)
    return model


if __name__ == "__main__":
    """Testing
    """
    cfg_shufflev2 = {
        'name': 'ShuffleNetV2',
        'min_sizes': [[16, 32], [64, 128], [256, 512]],
        'anchorNum_per_stage': 2,
        'steps': [8, 16, 32],
        'variance': [0.1, 0.2],
        'clip': False,
        'loc_weight': 2.0,
        'gpu_train': False,
        'batch_size': 8,
        'ngpu': 4,
        'epoch': 100,
        'pretrain': False,
        'ShuffleNetV2_return_layers': {'layer1': 1, 'layer2': 2, 'layer3': 3},
        'in_channel': 100,
        'out_channel': 128,
        'ShuffleNetV2': {
            'out_planes': [200, 400, 800],
            'stage_repeats': [4, 8, 4],
            'groups': 2,
            'image_size': [640, 640],
            'width_mult': 1.5,
            'n_class': 20
        }
    }

    model = ShuffleNetV2(cfg_shufflev2['ShuffleNetV2'])
    x = torch.randn(2, 3, 640, 640)
    y = model(x)
    print(y.shape)
    body = _utils.IntermediateLayerGetter(
        model, {'layer1': 1, 'layer2': 2, 'layer3': 3})

    out = (body(x))
    for i in out:
        print(out[i].shape)

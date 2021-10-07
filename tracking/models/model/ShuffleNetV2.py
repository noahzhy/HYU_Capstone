import math
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models._utils as _utils
from torch.autograd import Variable
from torch.nn import init


def conv_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True)
    )


def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True)
    )


def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.data.size()
    channels_per_group = num_channels // groups
    # reshape
    x = x.view(batchsize, groups, channels_per_group, height, width)
    x = torch.transpose(x, 1, 2).contiguous()
    # flatten
    x = x.view(batchsize, -1, height, width)
    return x


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, benchmodel):
        super(InvertedResidual, self).__init__()
        self.benchmodel = benchmodel
        self.stride = stride
        assert stride in [1, 2]

        oup_inc = oup//2

        if self.benchmodel == 1:
            #assert inp == oup_inc
            self.banch2 = nn.Sequential(
                # pw
                nn.Conv2d(oup_inc, oup_inc, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup_inc),
                nn.ReLU(inplace=True),
                # dw
                nn.Conv2d(oup_inc, oup_inc, 3, stride,
                          1, groups=oup_inc, bias=False),
                nn.BatchNorm2d(oup_inc),
                # pw-linear
                nn.Conv2d(oup_inc, oup_inc, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup_inc),
                nn.ReLU(inplace=True),
            )
        else:
            self.banch1 = nn.Sequential(
                # dw
                nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                # pw-linear
                nn.Conv2d(inp, oup_inc, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup_inc),
                nn.ReLU(inplace=True),
            )

            self.banch2 = nn.Sequential(
                # pw
                nn.Conv2d(inp, oup_inc, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup_inc),
                nn.ReLU(inplace=True),
                # dw
                nn.Conv2d(oup_inc, oup_inc, 3, stride,
                          1, groups=oup_inc, bias=False),
                nn.BatchNorm2d(oup_inc),
                # pw-linear
                nn.Conv2d(oup_inc, oup_inc, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup_inc),
                nn.ReLU(inplace=True),
            )

    @staticmethod
    def _concat(x, out):
        # concatenate along channel axis
        return torch.cat((x, out), 1)

    def forward(self, x):
        if 1 == self.benchmodel:
            x1 = x[:, :(x.shape[1]//2), :, :]
            x2 = x[:, (x.shape[1]//2):, :, :]
            out = self._concat(x1, self.banch2(x2))
        elif 2 == self.benchmodel:
            out = self._concat(self.banch1(x), self.banch2(x))

        return channel_shuffle(out, 2)


class ShuffleNetV2(nn.Module):
    def __init__(self, cfg):
        super(ShuffleNetV2, self).__init__()
        input_size = cfg['image_size'][0]
        self.stage_repeats = cfg['stage_repeats']
        width_mult = cfg['width_mult']
        n_class = cfg['n_class']

        assert input_size % 32 == 0

        self.stage_repeats = [4, 8, 4]
        # index 0 is invalid and should never be called.
        # only used for indexing convenience.
        if width_mult == 0.5:
            self.stage_out_channels = [-1, 24,  48,  96, 192, 1024]
        elif width_mult == 1.0:
            self.stage_out_channels = [-1, 24, 116, 232, 464, 1024]
        elif width_mult == 1.5:
            self.stage_out_channels = [-1, 24, 176, 352, 704, 1024]
        elif width_mult == 2.0:
            self.stage_out_channels = [-1, 24, 224, 488, 976, 2048]
        else:
            pass

        # building first layer
        input_channel = self.stage_out_channels[1]
        self.conv1 = conv_bn(3, input_channel, 2)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layers = []
        # building inverted residual blocks
        for idxstage in range(len(self.stage_repeats)):
            self.features = []
            numrepeat = self.stage_repeats[idxstage]
            output_channel = self.stage_out_channels[idxstage+2]
            for i in range(numrepeat):
                if i == 0:
                    # inp, oup, stride, benchmodel):
                    self.features.append(InvertedResidual(
                        input_channel, output_channel, 2, 2))
                else:
                    self.features.append(InvertedResidual(
                        input_channel, output_channel, 1, 1))
                input_channel = output_channel
            self.layers.append(self.features)

        # make it nn.Sequential
        self.layer1 = nn.Sequential(*self.layers[0])
        self.layer2 = nn.Sequential(*self.layers[1])
        self.layer3 = nn.Sequential(*self.layers[2])

        # building last several layers
        self.conv_last = conv_1x1_bn(
            input_channel, self.stage_out_channels[-1])
        self.globalpool = nn.Sequential(nn.AvgPool2d(int(input_size/32)))
        self.classifier = nn.Sequential(
            nn.Linear(self.stage_out_channels[-1], n_class))

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.conv_last(x)
        x = self.globalpool(x)
        x = x.view(-1, self.stage_out_channels[-1])
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
        'out_channel': 256,
        'ShuffleNetV2': {
            'out_planes': [200, 400, 800],
            'stage_repeats': [4, 8, 4],
            'groups': 2,
            'image_size': [640, 640],
            'width_mult': 1.5,  # 0.5 1.0 1.5, bug on 2.0
            'n_class': 20
        }
    }

    model = ShuffleNetV2(cfg_shufflev2['ShuffleNetV2'])
    print(model)
    x = torch.randn(1, 3, 640, 640)
    y = model(x)
    print(y.shape)
    body = _utils.IntermediateLayerGetter(
        model, {'layer1': 1, 'layer2': 2, 'layer3': 3})

    out = (body(x))
    for i in out:
        print(out[i].shape)

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


def BasicConv(inp, oup, kernel_size , stride , pad):
    return nn.Sequential(
        nn.Conv2d(inp, oup, kernel_size, stride, pad, bias=False),
        nn.BatchNorm2d(oup),
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


class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat(
            (torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)),
            dim=1,
        )


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class SpatialGate(nn.Module):
    def __init__(self):
        super(SpatialGate, self).__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = BasicConv(
            2,
            1,
            kernel_size,
            stride=1,
            pad=(kernel_size - 1) // 2,
        )

    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = F.sigmoid(x_out)
        return x * scale
    

class ChannelGate(nn.Module):
    def __init__(
        self, gate_channels, reduction_ratio=16, pool_types=["avg", "max"]
    ):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels),
        )
        self.pool_types = pool_types

    def forward(self, x):
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type == "avg":
                avg_pool = F.avg_pool2d(
                    x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3))
                )
                channel_att_raw = self.mlp(avg_pool)
            elif pool_type == "max":
                max_pool = F.max_pool2d(
                    x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3))
                )
                channel_att_raw = self.mlp(max_pool)
            elif pool_type == "lp":
                lp_pool = F.lp_pool2d(
                    x,
                    2,
                    (x.size(2), x.size(3)),
                    stride=(x.size(2), x.size * (3)),
                )
                channel_att_raw = self.mlp(lp_pool)
            elif pool_type == "lse":
                # LSE pool
                lse_pool = logsumexp_2d(x)
                channel_att_raw = self.mlp(lse_pool)

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw
        scale = (
            F.sigmoid(channel_att_sum).unsqueeze(2).unsqueeze(3).expand_as(x)
        )
        return x * scale


class CBAM(nn.Module):
    def __init__(
        self,
        gate_channels,
        reduction_ratio=16,
        pool_types=["avg", "max"],
        no_spatial=False,
    ):
        super(CBAM, self).__init__()
        self.ChannelGate = ChannelGate(
            gate_channels, reduction_ratio, pool_types
        )
        self.no_spatial = no_spatial
        if not no_spatial:
            self.SpatialGate = SpatialGate()

    def forward(self, x):
        x_out = self.ChannelGate(x)
        if not self.no_spatial:
            x_out = self.SpatialGate(x_out)
        return x_out


class ShuffleV2Block(nn.Module):
    def __init__(self, inp, oup, ksize, stride, benchmodel):
        super(ShuffleV2Block, self).__init__()
        self.benchmodel = benchmodel
        self.stride = stride
        assert stride in [1, 2]

        self.ksize = ksize
        pad = ksize // 2
        self.pad = pad
        # self.inp = inp
        oup_inc = oup//2

        if self.benchmodel == 1:
            #assert inp == oup_inc
        	self.banch2 = nn.Sequential(
                # pw
                nn.Conv2d(oup_inc, oup_inc, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup_inc),
                nn.ReLU(inplace=True),
                # dw
                nn.Conv2d(oup_inc, oup_inc, ksize, stride, pad, groups=oup_inc, bias=False),
                nn.BatchNorm2d(oup_inc),
                # pw-linear
                nn.Conv2d(oup_inc, oup_inc, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup_inc),
                # nn.ReLU(inplace=True),
                CBAM(gate_channels=oup_inc),
            )                
        else:                  
            self.banch1 = nn.Sequential(
                # dw
                nn.Conv2d(inp, inp, ksize, stride, pad, groups=inp, bias=False),
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
                nn.Conv2d(oup_inc, oup_inc, ksize, stride, pad, groups=oup_inc, bias=False),
                nn.BatchNorm2d(oup_inc),
                # pw-linear
                nn.Conv2d(oup_inc, oup_inc, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup_inc),
                # nn.ReLU(inplace=True),
                CBAM(gate_channels=oup_inc),
            )

    @staticmethod
    def _concat(x, out):
        # concatenate along channel axis
        return torch.cat((x, out), 1)        

    def forward(self, x):
        if 1==self.benchmodel:
            x1 = x[:, :(x.shape[1]//2), :, :]
            x2 = x[:, (x.shape[1]//2):, :, :]
            out = self._concat(x1, self.banch2(x2))
        elif 2==self.benchmodel:
            out = self._concat(self.banch1(x), self.banch2(x))

        return channel_shuffle(out, 2)


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

        # building last several layers
        # self.conv_last = conv_1x1_bn(input_channel, self.channels[-1])
        self.globalpool = nn.Sequential(nn.AvgPool2d(int(input_size/32)))
        self.classifier = nn.Sequential(nn.Linear(self.channels[-1], n_class))


    def _make_layer(self, num_layers, in_channels, out_channels, **kwargs):
        layers = []
        for i in range(num_layers):
            if i == 0:
                layers.append(ShuffleV2Block(in_channels, out_channels, 5, 2, 2))
            else:
                layers.append(ShuffleV2Block(in_channels, out_channels, 5, 1, 1))
            in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)

        c3 = self.layer1(x)
        c4 = self.layer2(c3)
        c5 = self.layer3(c4)

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

'''ShuffleNet in PyTorch.

See the paper "ShuffleNet: An Extremely Efficient Convolutional Neural Network for Mobile Devices" for more details.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
import torchvision.models._utils as _utils

class ShuffleBlock(nn.Module):
    def __init__(self, groups):
        super(ShuffleBlock, self).__init__()
        self.groups = groups

    def forward(self, x):
        '''Channel shuffle: [N,C,H,W] -> [N,g,C/g,H,W] -> [N,C/g,g,H,w] -> [N,C,H,W]'''
        N,C,H,W = x.size()
        g = self.groups
        # 维度变换之后必须要使用.contiguous()使得张量在内存连续之后才能调用view函数
        return x.view(N,g,int(C/g),H,W).permute(0,2,1,3,4).contiguous().view(N,C,H,W)

class Conv2D_BN_ReLU(nn.Module):
    def __init__(self,in_planes, out_planes,kernel_size=3, stride=2):
        super(Conv2D_BN_ReLU, self).__init__()

        self.conv = nn.Conv2d(in_planes, out_planes,
                               kernel_size=kernel_size, stride=stride, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        #self.maxpool=nn.Maxpool2d(kernel_size=kernel_size,stride=stride)
        #self.maxpool=nn.Sequential(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))


    def forward(self,x):
        out=self.conv(x)
        out=self.bn(out)
        out=F.relu(out)
        #out=self.maxpool(out)
        return F.relu(out)


class Bottleneck(nn.Module):
    def __init__(self, in_planes, out_planes, stride, groups):
        super(Bottleneck, self).__init__()
        self.stride = stride

        # bottleneck层中间层的channel数变为输出channel数的1/4
        mid_planes = int(out_planes/4)


        g = 1 if in_planes==24 else groups
        # 作者提到不在stage2的第一个pointwise层使用组卷积,因为输入channel数量太少,只有24
        self.conv1 = nn.Conv2d(in_planes, mid_planes,
                               kernel_size=1, groups=g, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_planes)
        self.shuffle1 = ShuffleBlock(groups=g)
        self.conv2 = nn.Conv2d(mid_planes, mid_planes,
                               kernel_size=3, stride=stride, padding=1,
                               groups=mid_planes, bias=False)
        self.bn2 = nn.BatchNorm2d(mid_planes)
        self.conv3 = nn.Conv2d(mid_planes, out_planes,
                               kernel_size=1, groups=groups, bias=False)
        self.bn3 = nn.BatchNorm2d(out_planes)

        self.shortcut = nn.Sequential()
        if stride == 2:
            self.shortcut = nn.Sequential(nn.AvgPool2d(3, stride=2, padding=1))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.shuffle1(out)
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        res = self.shortcut(x)
        out = F.relu(torch.cat([out,res], 1)) if self.stride==2 else F.relu(out+res)
        return out


class ShuffleNet(nn.Module):
    def __init__(self, cfg):
        super(ShuffleNet, self).__init__()
        out_planes = cfg['out_planes']
        num_blocks = cfg['num_blocks']
        groups = cfg['groups']


        self.conv1_bn_relu=Conv2D_BN_ReLU(3,24)
        self.pool1=nn.Sequential(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        self.in_planes = 24
        self.layer1 = self._make_layer(out_planes[0], num_blocks[0], groups)
        self.layer2 = self._make_layer(out_planes[1], num_blocks[1], groups)
        self.layer3 = self._make_layer(out_planes[2], num_blocks[2], groups)
        self.conv5_bn_relu = Conv2D_BN_ReLU(out_planes[2],out_planes[2], 1, 1)
        self.pool2=nn.Sequential(nn.AdaptiveAvgPool2d((1,1)))
        self.linear = nn.Linear(out_planes[2], 10)

    def _make_layer(self, out_planes, num_blocks, groups):
        layers = []
        for i in range(num_blocks):
            if i == 0:
                layers.append(Bottleneck(self.in_planes,
                                         out_planes-self.in_planes,
                                         stride=2, groups=groups))
            else:
                layers.append(Bottleneck(self.in_planes,
                                         out_planes,
                                         stride=1, groups=groups))
            self.in_planes = out_planes
        return nn.Sequential(*layers)

    def forward(self, x):

        #out = F.relu(self.bn1(self.conv1(x)))
        out=self.conv1_bn_relu(x)
        out=self.pool1(out)

        out = self.layer1(out)

        out = self.layer2(out)

        out = self.layer3(out)

        #out = F.avg_pool2d(out, 4)
        out=self.conv5_bn_relu(out)
        out=self.pool2(out)
        out = out.view(out.size(0), -1)

        out = self.linear(out)
        return out


def ShuffleNetG2(cfg):

    return ShuffleNet(cfg)

def ShuffleNetG3():
    cfg = {
        'out_planes': [240,480,960],
        'num_blocks': [4,8,4],
        'groups': 3
    }
    return ShuffleNet(cfg)


def test():
    cfg = {
        'out_planes': [200, 400, 800],
        'num_blocks': [4, 8, 4],
        'groups': 2
    }
    net = ShuffleNetG2(cfg)
    x = torch.randn(1,3,224,224)
    y = net(x)
    print(y)
    x = torch.randn(1, 3, 224, 224)
    body = _utils.IntermediateLayerGetter(net, {'layer1':1,'layer2':2,'layer3':3})

    out=(body(x))
    for i in out:
        print(out[i].shape)





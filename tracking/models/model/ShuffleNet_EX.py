import sys
sys.path.append('../')
import torch
import torch.nn as nn
import torchvision.models._utils as _utils
from models.model.blocks import *


class ShuffleNetV2_EX(nn.Module):
    def __init__(self, input_size=640, n_class=12):
        super(ShuffleNetV2_EX, self).__init__()

        assert input_size % 32 == 0
        architecture_txt = [
            'Shuffle3', 'Shuffle3', 'Xception', 'Shuffle5',

            'Shuffle5', 'Shuffle5', 'Shuffle3', 'Shuffle3', # stage2
            'Shuffle7', 'Shuffle3', 'Shuffle7', 'Shuffle5', 'Shuffle5', 'Shuffle3', 'Shuffle7', 'Shuffle3', # stage3
            'Shuffle7', 'Shuffle5', 'Xception', 'Shuffle7' # stage4
        ]

        self.stage_repeats = [4, 4, 8, 4]
        self.stage_out_channels = [-1, 16, 48, 128, 256, 512, 1280]

        input_channel = self.stage_out_channels[1]
        self.first_conv = nn.Sequential(
            nn.Conv2d(3, input_channel, 3, 2, 1, bias=False),
            nn.BatchNorm2d(input_channel),
            HS(),
        )

        # self.features = []
        self.layers = []
        archIndex = 0
        for idxstage in range(len(self.stage_repeats)):
            numrepeat = self.stage_repeats[idxstage]
            output_channel = self.stage_out_channels[idxstage+2]
            activation = 'HS' if idxstage >= 1 else 'ReLU'
            useSE = 'True' if idxstage >= 2 else False
            self.features = []
            for i in range(numrepeat):
                if i == 0:
                    inp, outp, stride = input_channel, output_channel, 2
                else:
                    inp, outp, stride = input_channel // 2, output_channel, 1

                blockIndex = architecture_txt[archIndex]
                archIndex += 1
                if blockIndex == 'Shuffle3':
                    self.features.append(Shufflenet(inp, outp, base_mid_channels=outp // 2, ksize=3, stride=stride,
                                    activation=activation, useSE=useSE))
                elif blockIndex == 'Shuffle5':
                    self.features.append(Shufflenet(inp, outp, base_mid_channels=outp // 2, ksize=5, stride=stride,
                                    activation=activation, useSE=useSE))
                elif blockIndex == 'Shuffle7':
                    self.features.append(Shufflenet(inp, outp, base_mid_channels=outp // 2, ksize=7, stride=stride,
                                    activation=activation, useSE=useSE))
                elif blockIndex == 'Xception':
                    self.features.append(Xception(inp, outp, base_mid_channels=outp // 2, stride=stride,
                                    activation=activation, useSE=useSE))
                else:
                    raise NotImplementedError
                input_channel = output_channel
            self.layers.append(self.features)

        self.layer0 = nn.Sequential(*self.layers[0])
        self.layer1 = nn.Sequential(*self.layers[1]) # stage2
        self.layer2 = nn.Sequential(*self.layers[2]) # stage3
        self.layer3 = nn.Sequential(*self.layers[3]) # stage4

        self.conv_last = nn.Sequential(
            nn.Conv2d(input_channel, 1280, 1, 1, 0, bias=False),
            nn.BatchNorm2d(1280),
            HS()
        )
        self.globalpool = nn.AvgPool2d(7)
        self.SE = SELayer(1280)
        self.fc = nn.Sequential(
            nn.Linear(1280, 1280, bias=False),
            HS(),
        )
        self.dropout = nn.Dropout(0.2)
        self.classifier = nn.Sequential(nn.Linear(1280, n_class, bias=False))
        self._initialize_weights()

    def forward(self, x):
        x = self.first_conv(x)
        # x = self.features(x)

        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.conv_last(x)

        x = self.globalpool(x)
        x = self.SE(x)

        x = x.contiguous().view(-1, 1280)

        x = self.fc(x)
        x = self.dropout(x)
        x = self.classifier(x)
        return x


    def _initialize_weights(self):
        for name, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                if 'first' in name or 'SE' in name:
                    nn.init.normal_(m.weight, 0, 0.01)
                else:
                    nn.init.normal_(m.weight, 0, 1.0 / m.weight.shape[1])
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0001)
                nn.init.constant_(m.running_mean, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0001)
                nn.init.constant_(m.running_mean, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


if __name__ == "__main__":
    model = ShuffleNetV2_EX()
    # print(model)

    x = torch.rand(2, 3, 640, 640)
    y = model(x)
    print(y.shape)
    body = _utils.IntermediateLayerGetter(
        model, {'layer1': 1, 'layer2': 2, 'layer3': 3})

    out = (body(x))
    for i in out:
        print(out[i].shape)

    # print(test_outputs.size())

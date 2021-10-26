import math
import os
from collections import OrderedDict

import torch
import torch.nn as nn
from models.neck.neck import conv_bn, conv_bn1X1


def head_cls_shared(inp, oup, stride = 1, leaky = 0, repeat = 2):
    peranchor_feature = nn.ModuleList()
    for j in range(repeat):
        peranchor_feature.append(conv_bn(inp,oup,stride,leaky))
    nn.Sequential(*peranchor_feature)
    return nn.Sequential(*peranchor_feature)

def head_loc_shared(inp, oup, stride = 1, leaky = 0, repeat = 2):
    peranchor_feature = nn.ModuleList()
    for j in range(repeat):
        peranchor_feature.append(conv_bn(inp,oup,stride,leaky))
    nn.Sequential(*peranchor_feature)
    return nn.Sequential(*peranchor_feature)

def head_emb_shared(inp, oup, stride = 1, leaky = 0, repeat = 2):
    peranchor_feature = nn.ModuleList()
    for j in range(repeat):
        peranchor_feature.append(conv_bn1X1(inp,oup,stride,leaky))
    nn.Sequential(*peranchor_feature)
    return nn.Sequential(*peranchor_feature)

def cls_head(inp, oup = 12, fpnNum = 3, anchorNum = 2):
    cls_heads = nn.ModuleList()
    for i in range(fpnNum):
        for j in range(anchorNum):
            cls_heads.append(
                nn.Conv2d(inp, oup, kernel_size=(1, 1), stride=1, padding=0),
            )
    return cls_heads

def loc_head(inp, oup = 4, fpnNum = 3, anchorNum = 2):
    bbox_heads = nn.ModuleList()
    for i in range(fpnNum):
        for j in range(anchorNum):
            bbox_heads.append(
                nn.Conv2d(inp, oup, kernel_size=(1, 1), stride=1, padding=0)
            )
    return bbox_heads

def emb_head(inp, oup = 128, fpnNum = 3, anchorNum = 2):
    emb_heads = nn.ModuleList()
    for i in range(fpnNum):
        for j in range(anchorNum):
            emb_heads.append(
                nn.Conv2d(inp, oup, kernel_size=(1, 1), stride=1, padding=0)
            )
    return emb_heads

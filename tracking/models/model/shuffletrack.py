import sys
sys.path.append('./')
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models._utils as _utils
import torchvision.models.detection.backbone_utils as backbone_utils
from config.config import cfg_re50, cfg_shuffle, cfg_shufflev2
from models.head.head import (make_cls_head, make_emb_head, make_loc_head,
                              task_specific_cls, task_specific_emb,
                              task_specific_loc)
from models.model.ShuffleNet import ShuffleNetG2
from models.model.ShuffleNetV2 import ShuffleNetV2
from models.neck.neck import FPN as FPN
from models.neck.neck import SSH as SSH
from models.neck.neck import task_shared


class ShuffleTrackNet(nn.Module):

    def __init__(self, cfg=None, phase='train'):
        """
        :param cfg:  Network related settings.
        :param phase: train or test.
        """
        super(ShuffleTrackNet, self).__init__()
        self.phase = phase
        backbone = None
        self.n_class = cfg['n_class']

        if cfg['name'] == 'Resnet50':
            import torchvision.models as models
            backbone = models.resnet50(pretrained=cfg['pretrain'])
            self.body = _utils.IntermediateLayerGetter(
                backbone, cfg['return_layers'])
            in_channels_stage2 = cfg['in_channel']
            in_channels_list = [
                in_channels_stage2 * 2,
                in_channels_stage2 * 4,
                in_channels_stage2 * 8,
            ]
        elif cfg['name'] == 'ShuffleNetG2':
            backbone = ShuffleNetG2(cfg['ShuffleNetG2'])
            self.body = _utils.IntermediateLayerGetter(
                backbone, cfg['ShuffleNetG2_return_layers'])
            pass
        elif cfg['name'] == 'ShuffleNetV2':
            backbone = ShuffleNetV2(cfg['ShuffleNetV2'])
            self.body = _utils.IntermediateLayerGetter(
                backbone, cfg['ShuffleNetV2_return_layers'])
            # ch_list = [24, 132, 264, 528]
            # cfg['in_channel'] = ch_list[int(cfg['ShuffleNetV2']*2-1)]
            in_channels_list = [132, 264, 528]
            pass

        # not total anchor,indicate per stage anchors
        anchorNum = cfg['anchorNum_per_stage']
        print('in_channels_list', in_channels_list)
        out_channels = cfg['out_channel']
        self.coco = cfg['coco']
        self.fpn = FPN(in_channels_list, out_channels)
        self.ssh1 = SSH(out_channels, out_channels)
        self.ssh2 = SSH(out_channels, out_channels)
        self.ssh3 = SSH(out_channels, out_channels)
        # task shared
        self.task_shared = task_shared(
            out_channels, out_channels, anchorNum=anchorNum)
        # task specific
        self.cls_task = task_specific_cls(out_channels, out_channels)
        self.loc_task = task_specific_loc(out_channels, out_channels)
        self.emb_task = task_specific_emb(out_channels, out_channels)
        # head
        self.cls_heads = make_cls_head(inp=out_channels, fpnNum=len(
            in_channels_list), anchorNum=anchorNum)
        self.loc_heads = make_loc_head(inp=out_channels, fpnNum=len(
            in_channels_list), anchorNum=anchorNum)
        self.emb_heads = make_emb_head(inp=out_channels, fpnNum=len(
        in_channels_list), anchorNum=anchorNum)
        # classifier
        self.classifier = nn.Linear(128, 547)

    def forward(self, inputs):
        out = self.body(inputs)
        # FPN
        fpn = self.fpn(out)
        # SSH
        feature1 = self.ssh1(fpn[0])
        feature2 = self.ssh2(fpn[1])
        feature3 = self.ssh3(fpn[2])
        fpnfeatures = [feature1, feature2, feature3]
        features = []
        # task shared
        for fpnfeature in fpnfeatures:
            per_anchor_feature = []
            for per_task_shared in self.task_shared:
                per_anchor_feature.append(per_task_shared(fpnfeature))
            features.append(per_anchor_feature)
        # task specific
        cls_heads, loc_heads, emb_heads = [], [], []
        for i, per_fpn_features in enumerate(features):
            for j, per_anchor_feature in enumerate(per_fpn_features):
                cls_task_feature = self.cls_task(per_anchor_feature)
                loc_task_feature = self.loc_task(per_anchor_feature)
                emb_task_feature = self.emb_task(per_anchor_feature)
                # cls feature,only one class but with background total class is two
                cls_head = self.cls_heads[i * len(per_fpn_features) + j](cls_task_feature)
                loc_head = self.loc_heads[i * len(per_fpn_features) + j](loc_task_feature)
                emb_head = self.emb_heads[i * len(per_fpn_features) + j](emb_task_feature)
                # loc frature,(x,y,w,h)
                cls_head = cls_head.permute(0, 2, 3, 1).contiguous().view(cls_head.shape[0], -1, 12)
                loc_head = loc_head.permute(0, 2, 3, 1).contiguous().view(loc_head.shape[0], -1, 4)
                emb_head = emb_head.permute(0, 2, 3, 1).contiguous().view(emb_head.shape[0], -1, 128)
                # emb feature with 128 dim
                cls_heads.append(cls_head)
                loc_heads.append(loc_head)
                emb_heads.append(emb_head)


        bbox_regressions = torch.cat(
            [feature for i, feature in enumerate(loc_heads)], dim=1)
        classifications = torch.cat(
            [feature for i, feature in enumerate(cls_heads)], dim=1)
        emb_features = torch.cat(
            [feature for i, feature in enumerate(emb_heads)], dim=1)
        classifier = self.classifier(emb_features)
        return [bbox_regressions, classifications, classifier]


if __name__ == '__main__':
    cfg = cfg_shufflev2
    # cfg = cfg_re50
    model = ShuffleTrackNet(cfg=cfg)
    inpunt = torch.randn(5, 3, 640, 640)
    cls_heads, loc_heads, emb_heads = model(inpunt)
    # cls_heads, loc_heads = model(inpunt)
    # print(cls_heads.shape, loc_heads.shape, emb_heads.shape)
    print(model)

    from ptflops import get_model_complexity_info
    macs, params = get_model_complexity_info(model, (3, 640, 640), as_strings=True,
                                            print_per_layer_stat=True, verbose=True)
    print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    print('{:<30}  {:<8}'.format('Number of parameters: ', params))

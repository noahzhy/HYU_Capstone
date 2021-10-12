import argparse
import os
import parser
import time
from operator import mod

import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from config.config import cfg_re50, cfg_shuffle, cfg_shufflev2
from dataloader import Gtdataset
from models.model.shuffletrack import ShuffleTrackNet
from utils.box_utils import decode
from utils.multibox_loss import MultiBoxLoss
from utils.prior_box import PriorBox


CFG = cfg_shufflev2
# CFG = cfg_re50
nms_threshold = 0.4
vis_thres = 0.6


def train(model, train_loader):
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4)
    criterion = MultiBoxLoss(2, 0.35, True, 0, True, 547, 0.35, False)
    for epoch in range(1, CFG['epoch']+1):
        tqdm_train = tqdm(train_loader)
        for img, target in tqdm_train:
            img = img.cuda()
            target = target.cuda()
            outputs = model(img)
            priorbox = PriorBox(CFG)
            with torch.no_grad():
                priors = priorbox.forward()
                priors = priors.cuda()
            loss_l, loss_c, loss_id = criterion(outputs, priors, target)
            loss = loss_l + loss_c + loss_id
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            tqdm_train.set_description(
                f'epoch: {epoch}, total loss: {loss.item()}')
        state = {'net': model.state_dict(), 'epoch': epoch}
        torch.save(state, 'epoch_{}_loss_{}.pth'.format(epoch, loss.item()))


if __name__ == '__main__':
    gtdataset = Gtdataset()
    trainloader = DataLoader(gtdataset, batch_size=16, shuffle=True)
    model = ShuffleTrackNet(cfg=CFG).cuda()
    # print(model)
    # model = torch.nn.DataParallel(model)
    # check = torch.load("epoch_16_loss_0.09874087572097778.pth")
    # model.load_state_dict(check["net"])
    train(model, trainloader)

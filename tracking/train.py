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
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import *
from tqdm import tqdm

from config.config import *
from dataloader import *
from models.model.shuffletrack import ShuffleTrackNet
from utils.box_utils import decode
from utils.prior_box import PriorBox
from utils.multibox_loss import MultiBoxLoss
from utils.augmentations import SSDAugmentation


CFG = cfg_shufflev2
epochs = CFG['epoch']
# torch.cuda.set_device(3)


def train(model, train_loader):
    optimizer = optim.SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=5e-4)
    # scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5, eta_min=1e-5)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[
            int(0.1 * epochs),
            int(0.5 * epochs),
            int(0.75 * epochs)
        ],
        gamma=0.1
    )
    criterion = MultiBoxLoss(12, 0.3, True, 0, True, 5, 0.3, False)

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
            # loss = loss_l + loss_c
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            tqdm_train.set_description(
                f'epoch: {epoch}, total loss: {loss.item()}')
        state = {'net': model.state_dict(), 'epoch': epoch}
        torch.save(state, 'weights/epoch_{}_loss_{}.pth'.format(epoch, loss.item()))

        scheduler.step()


if __name__ == '__main__':
    gtds = Gtdataset()
    # gtds = Detdataset()
    trainloader = DataLoader(gtds, batch_size=32, shuffle=True)
    model = ShuffleTrackNet(cfg=CFG).cuda()
    # print(model)
    # torch.distributed.init_process_group()
    # model = torch.nn.DataParallel(model)
    # check = torch.load("epoch_77_loss_2.0705957412719727.pth")
    # model.load_state_dict(check["net"])
    train(model, trainloader)

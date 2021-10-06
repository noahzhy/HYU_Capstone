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
from dataloader import Detdataset, Gtdataset, Shuffledataset
from models.model.retinatrack import RetinaTrackNet
from utils.box_utils import decode
from utils.multibox_loss import MultiBoxLoss
from utils.prior_box import PriorBox

from ptflops import get_model_complexity_info


CFG = cfg_shufflev2
nms_threshold = 0.4
vis_thres = 0.6
# torch.cuda.set_device(5)


def predict(model, img_path, save_path):
    cfg = CFG
    img_raw = cv2.imread(img_path)
    img = cv2.resize(img_raw, (640, 640))
    _, im_height, im_width = img.shape
    img = img.transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).unsqueeze(0).cuda()
    img = img.cuda().float() / 255.0
    scale = torch.Tensor([img_raw.shape[1], img_raw.shape[0],
                         img_raw.shape[1], img_raw.shape[0]]).cuda()
    model.eval()
    loc, conf, classifier = model(img)  # forward pass
    priorbox = PriorBox(cfg)
    with torch.no_grad():
        priors = priorbox.forward()
        priors = priors.cuda()
    with torch.no_grad():
        boxes = decode(loc.squeeze(0), priors, cfg['variance'])
        boxes = boxes * scale
        conf = F.softmax(conf, dim=-1)
        scores = conf.squeeze(0).cpu().numpy()[:, 1]
        inds = np.where(scores > 0.6)[0]
        boxes = boxes[inds]
        scores = scores[inds]
        scores = torch.from_numpy(scores).cuda().unsqueeze(1)
        classifier = F.softmax(classifier, dim=-1).squeeze(0)
        classifier = classifier.data.max(-1, keepdim=True)[1]
        classifier = classifier[inds].float()
        dets = torch.cat((boxes, scores, classifier), 1)
        i = torchvision.ops.boxes.nms(dets[:, :4], dets[:, 5], nms_threshold)
        dets = dets[i]
    for b in dets:
        print(b)
        if b[4] < vis_thres:
            continue
        text = "car: {:d}".format(int(b[5]))
        print(b)
        b = list(map(int, b))
        cv2.rectangle(img_raw, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 2)
        cx = b[0]
        cy = b[1] + 12
        cv2.putText(img_raw, text, (cx, cy),
                    cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))
    cv2.imwrite(save_path, img_raw)


def train(model, train_loader):
    optimizer = optim.SGD(model.parameters(), lr=0.001,
                          momentum=0.9, weight_decay=5e-4)
    criterion = MultiBoxLoss(2, 0.35, True, 0, True, 7, 0.35, False)
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
        torch.save(state, 'test.pth')


if __name__ == '__main__':
    model = RetinaTrackNet(cfg=CFG).cuda()
    macs, params = get_model_complexity_info(model, (3, 640, 640), as_strings=True,
                                            print_per_layer_stat=True, verbose=True)
    print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    print('{:<30}  {:<8}'.format('Number of parameters: ', params))
    # print(model)
    quit()
    # test = Gtdataset()
    # trainloader = DataLoader(test, batch_size=8, shuffle=True)
    # train(model, trainloader)
    # predict(model, 'source/test.jpg', 'source/result_img.png')
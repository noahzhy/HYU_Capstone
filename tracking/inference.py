from __future__ import print_function

import argparse
import parser
import time

import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torchvision

from config.config import cfg_re50, cfg_shuffle, cfg_shufflev2
from models.model.shuffletrack import ShuffleTrackNet
from utils.box_utils import decode
from utils.prior_box import PriorBox


parser = argparse.ArgumentParser()
parser.add_argument('--trained_model', default='../epoch_21_loss_1.8753526210784912.pth',
                    type=str, help='Trained state_dict file path to open')
parser.add_argument('--confidence_threshold', default=0.1,
                    type=float, help='confidence_threshold')
parser.add_argument('--nms_threshold', default=0.4,
                    type=float, help='nms_threshold')
parser.add_argument('--vis_thres', default=0.15, type=float,
                    help='visualization_threshold')
parser.add_argument('--image', default='images/000011.jpg',
                    help='test image path')
parser.add_argument('--save_path',
                    default='images/result_img_01.jpg',
                    help='test image path')
args = parser.parse_args()


if __name__ == '__main__':
    cfg = cfg_shufflev2
    model = ShuffleTrackNet(cfg=cfg).cuda()
    check = torch.load(args.trained_model)
    model.load_state_dict(check["net"], False)
    model.eval()
    img_raw = cv2.imread(args.image)
    img = cv2.resize(img_raw,(640,640))
    _, im_height, im_width = img.shape
    img = img.transpose(2, 0, 1)
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).unsqueeze(0).cuda()
    img = img.cuda().float() / 255.0
    scale = torch.Tensor([img_raw.shape[1], img_raw.shape[0], img_raw.shape[1], img_raw.shape[0]]).cuda()
    
    # for i in range(10):
    tic = time.time()
    loc, conf, classifier = model(img)  # forward pass
    toc = time.time()
        # print((toc-tic)*1000)

    priorbox = PriorBox(cfg)
    with torch.no_grad():
        priors = priorbox.forward()
        priors = priors.cuda()
        boxes = decode(loc.squeeze(0), priors, cfg['variance'])
        boxes = boxes * scale
        conf = F.softmax(conf, dim=-1)
        scores = conf.squeeze(0).cpu().numpy()[:, 1]
        inds = np.where(scores > args.confidence_threshold)[0]
        boxes = boxes[inds]
        scores = scores[inds]
        scores = torch.from_numpy(scores).cuda().unsqueeze(1)
        classifier = F.softmax(classifier, dim=-1).squeeze(0)
        classifier = classifier.data.max(-1, keepdim=True)[1]
        classifier = classifier[inds].float()
        dets = torch.cat((boxes,scores,classifier),1)
        i = torchvision.ops.boxes.nms(dets[:,:4], dets[:,5], args.nms_threshold)
        dets = dets[i]
    for b in dets:
        print(list(map(float, b)))
        if b[4] < args.vis_thres:
            continue
        text = "id: {:d} conf: {:.2f}".format(int(b[5]), b[4])
        # print(text)
        b = list(map(int, b))
        cv2.rectangle(img_raw, (b[0], b[1]), (b[2], b[3]), (0, 255, 0), 2)
        cx = b[0]
        cy = b[1]
        cv2.putText(img_raw, text, (cx, cy),
                    cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255))
    cv2.imwrite(args.save_path, img_raw)

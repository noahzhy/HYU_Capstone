import argparse
import torch
import torch.backends.cudnn as cudnn
import numpy as np
from utils.prior_box import PriorBox
import cv2
from models.model.shuffletrack import ShuffleTrackNet
from config.config import cfg_re50, cfg_shuffle, cfg_shufflev2
from utils.box_utils import decode
import time
import torchvision
import parser
import torch.nn.functional as F


parser = argparse.ArgumentParser()
parser.add_argument('--trained_model', default='../epoch_7_loss_1.2427860498428345.pth',
                    type=str, help='Trained state_dict file path to open')
parser.add_argument('--confidence_threshold', default=0.6,
                    type=float, help='confidence_threshold')
parser.add_argument('--nms_threshold', default=0.4,
                    type=float, help='nms_threshold')
parser.add_argument('--vis_thres', default=0.8, type=float,
                    help='visualization_threshold')
parser.add_argument('--image', default='images/000001.jpg',
                    help='test image path')
args = parser.parse_args()


if __name__ == '__main__':
    cfg = cfg_shufflev2
    # cfg = cfg_re50
    model = ShuffleTrackNet(cfg=cfg).cuda()
    check = torch.load(args.trained_model)
    model.load_state_dict(check["net"])
    model.eval()
    img_raw = cv2.imread(args.image)
    img = cv2.resize(img_raw, (640, 640))
    img = img.transpose(2, 0, 1)
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).unsqueeze(0).float() / 255.0
    img = img.cuda()
    for i in range(20):
        with torch.no_grad():
            tic = time.time()
            loc, conf, classifier = model(img)  # forward pass
            toc = time.time()
            print("inference time: {}ms".format((toc-tic)*1000))
    scale = torch.Tensor([img_raw.shape[1], img_raw.shape[0],
                         img_raw.shape[1], img_raw.shape[0]]).cuda()
    print(scale)

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
        print(boxes, inds)
        scores = torch.from_numpy(scores).cuda().unsqueeze(1)
        classifier = F.softmax(classifier, dim=-1).squeeze(0)
        classifier = classifier.data.max(-1, keepdim=True)[1]
        classifier = classifier[inds].float()
        dets = torch.cat((boxes, scores, classifier), 1)
        i = torchvision.ops.boxes.nms(
            dets[:, :4], dets[:, 5], args.nms_threshold)
        dets = dets[i]

    for b in dets:
        # print("c:", b[4])
        if b[4] < args.vis_thres:
            continue
        text = "id: {:d}".format(int(b[5]))
        # print(text)
        b = list(map(int, b))
        print(b)
        cv2.rectangle(img_raw, (b[0], b[1]), (b[2], b[3]), (0, 255, 0), 2)
        cx = b[0]
        cy = b[1]
        cv2.putText(img_raw, text, (cx, cy),
                    cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))
    cv2.imwrite('images/result_img.jpg', img_raw)

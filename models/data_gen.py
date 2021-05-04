import cv2
import math
import numpy as np
import tensorflow as tf
import keras.backend as K
from PIL import Image
from random import shuffle
from matplotlib.colors import hsv_to_rgb, rgb_to_hsv


def draw_gaussian(heatmap, center, radius, k=1):
    diameter = 2 * radius + 1
    gaussian = gaussian2D((diameter, diameter), sigma=diameter / 6)

    x, y = int(center[0]), int(center[1])

    height, width = heatmap.shape[0:2]

    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)

    masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
    masked_gaussian = gaussian[radius - top:radius +
                               bottom, radius - left:radius + right]
    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:
        np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
    return heatmap


def gaussian_radius(det_size, min_overlap=0.7):
    height, width = det_size

    a1 = 1
    b1 = (height + width)
    c1 = width * height * (1 - min_overlap) / (1 + min_overlap)
    sq1 = np.sqrt(b1 ** 2 - 4 * a1 * c1)
    r1 = (b1 + sq1) / 2

    a2 = 4
    b2 = 2 * (height + width)
    c2 = (1 - min_overlap) * width * height
    sq2 = np.sqrt(b2 ** 2 - 4 * a2 * c2)
    r2 = (b2 + sq2) / 2

    a3 = 4 * min_overlap
    b3 = -2 * min_overlap * (height + width)
    c3 = (min_overlap - 1) * width * height
    sq3 = np.sqrt(b3 ** 2 - 4 * a3 * c3)
    r3 = (b3 + sq3) / 2
    return min(r1, r2, r3)


class Generator(object):
    def __init__(self, batch_size, train_lines, val_lines,
                 input_size, num_classes, max_objects=100):

        self.batch_size = batch_size
        self.train_lines = train_lines
        self.val_lines = val_lines
        self.input_size = input_size
        self.output_size = (int(input_size[0]/4), int(input_size[1]/4))
        self.num_classes = num_classes
        self.max_objects = max_objects

    def get_random_data(self, annotation_line, input_shape, jitter=.3, hue=.1, sat=1.5, val=1.5, random=True):
        line = annotation_line.split()
        image = Image.open(line[0])
        iw, ih = image.size
        h, w = input_shape
        box = np.array([np.array(list(map(int, box.split(','))))
                        for box in line[1:]])

        if not random:
            # resize image
            scale = min(w/iw, h/ih)
            nw = int(iw*scale)
            nh = int(ih*scale)
            dx = (w-nw)//2
            dy = (h-nh)//2

            image = image.resize((nw, nh), Image.BICUBIC)
            new_image = Image.new('RGB', (w, h), (128, 128, 128))
            new_image.paste(image, (dx, dy))
            image_data = np.array(new_image, np.float32)

            # correct boxes
            box_data = np.zeros((len(box), 5))
            if len(box) > 0:
                np.random.shuffle(box)
                box[:, [0, 2]] = box[:, [0, 2]]*nw/iw + dx
                box[:, [1, 3]] = box[:, [1, 3]]*nh/ih + dy
                box[:, 0:2][box[:, 0:2] < 0] = 0
                box[:, 2][box[:, 2] > w] = w
                box[:, 3][box[:, 3] > h] = h
                box_w = box[:, 2] - box[:, 0]
                box_h = box[:, 3] - box[:, 1]
                box = box[np.logical_and(box_w > 1, box_h > 1)]
                box_data = np.zeros((len(box), 5))
                box_data[:len(box)] = box

            return image_data, box_data

        # resize image
        new_ar = w/h * rand(1-jitter, 1+jitter)/rand(1-jitter, 1+jitter)
        scale = rand(0.25, 2)
        if new_ar < 1:
            nh = int(scale*h)
            nw = int(nh*new_ar)
        else:
            nw = int(scale*w)
            nh = int(nw/new_ar)
        image = image.resize((nw, nh), Image.BICUBIC)

        # place image
        dx = int(rand(0, w-nw))
        dy = int(rand(0, h-nh))
        new_image = Image.new('RGB', (w, h), (128, 128, 128))
        new_image.paste(image, (dx, dy))
        image = new_image

        # flip image or not
        flip = rand() < .5
        if flip:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)

        # distort image
        hue = rand(-hue, hue)
        sat = rand(1, sat) if rand() < .5 else 1/rand(1, sat)
        val = rand(1, val) if rand() < .5 else 1/rand(1, val)
        x = cv2.cvtColor(np.array(image, np.float32)/255, cv2.COLOR_RGB2HSV)
        x[..., 0] += hue*360
        x[..., 0][x[..., 0] > 1] -= 1
        x[..., 0][x[..., 0] < 0] += 1
        x[..., 1] *= sat
        x[..., 2] *= val
        x[x[:, :, 0] > 360, 0] = 360
        x[:, :, 1:][x[:, :, 1:] > 1] = 1
        x[x < 0] = 0
        image_data = cv2.cvtColor(x, cv2.COLOR_HSV2RGB)*255

        # correct boxes
        box_data = np.zeros((len(box), 5))
        if len(box) > 0:
            np.random.shuffle(box)
            box[:, [0, 2]] = box[:, [0, 2]]*nw/iw + dx
            box[:, [1, 3]] = box[:, [1, 3]]*nh/ih + dy
            if flip:
                box[:, [0, 2]] = w - box[:, [2, 0]]
            box[:, 0:2][box[:, 0:2] < 0] = 0
            box[:, 2][box[:, 2] > w] = w
            box[:, 3][box[:, 3] > h] = h
            box_w = box[:, 2] - box[:, 0]
            box_h = box[:, 3] - box[:, 1]
            # discard invalid box
            box = box[np.logical_and(box_w > 1, box_h > 1)]
            box_data = np.zeros((len(box), 5))
            box_data[:len(box)] = box

        return image_data, box_data

    def generate(self, train=True):
        while True:
            if train:
                shuffle(self.train_lines)
                lines = self.train_lines
            else:
                shuffle(self.val_lines)
                lines = self.val_lines

            batch_images = np.zeros(
                (self.batch_size, self.input_size[0], self.input_size[1], self.input_size[2]), dtype=np.float32)
            batch_hms = np.zeros(
                (self.batch_size, self.output_size[0], self.output_size[1], self.num_classes), dtype=np.float32)
            batch_whs = np.zeros(
                (self.batch_size, self.max_objects, 2), dtype=np.float32)
            batch_regs = np.zeros(
                (self.batch_size, self.max_objects, 2), dtype=np.float32)
            batch_reg_masks = np.zeros(
                (self.batch_size, self.max_objects), dtype=np.float32)
            batch_indices = np.zeros(
                (self.batch_size, self.max_objects), dtype=np.float32)

            b = 0
            for annotation_line in lines:
                img, y = self.get_random_data(
                    annotation_line, self.input_size[0:2], random=train)

                if len(y) != 0:
                    boxes = np.array(y[:, :4], dtype=np.float32)
                    boxes[:, 0] = boxes[:, 0] / \
                        self.input_size[1]*self.output_size[1]
                    boxes[:, 1] = boxes[:, 1] / \
                        self.input_size[0]*self.output_size[0]
                    boxes[:, 2] = boxes[:, 2] / \
                        self.input_size[1]*self.output_size[1]
                    boxes[:, 3] = boxes[:, 3] / \
                        self.input_size[0]*self.output_size[0]

                for i in range(len(y)):
                    bbox = boxes[i].copy()
                    bbox = np.array(bbox)
                    bbox[[0, 2]] = np.clip(
                        bbox[[0, 2]], 0, self.output_size[1] - 1)
                    bbox[[1, 3]] = np.clip(
                        bbox[[1, 3]], 0, self.output_size[0] - 1)
                    cls_id = int(y[i, -1])

                    h, w = bbox[3] - bbox[1], bbox[2] - bbox[0]
                    if h > 0 and w > 0:
                        ct = np.array(
                            [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2], dtype=np.float32)
                        ct_int = ct.astype(np.int32)

                        # get heatmap
                        radius = gaussian_radius((math.ceil(h), math.ceil(w)))
                        radius = max(0, int(radius))
                        batch_hms[b, :, :, cls_id] = draw_gaussian(
                            batch_hms[b, :, :, cls_id], ct_int, radius)

                        batch_whs[b, i] = 1. * w, 1. * h
                        # offset
                        batch_regs[b, i] = ct - ct_int
                        # set mask as 1
                        batch_reg_masks[b, i] = 1

                        batch_indices[b, i] = ct_int[1] * \
                            self.output_size[0] + ct_int[0]

                # RGB -> BGR
                img = np.array(img, dtype=np.float32)[:, :, ::-1]
                batch_images[b] = preprocess_image(img)
                b = b + 1
                if b == self.batch_size:
                    b = 0
                    yield [batch_images, batch_hms, batch_whs, batch_regs, batch_reg_masks, batch_indices], np.zeros((self.batch_size,))

                    batch_images = np.zeros(
                        (self.batch_size, self.input_size[0], self.input_size[1], 3), dtype=np.float32)

                    batch_hms = np.zeros((self.batch_size, self.output_size[0], self.output_size[1], self.num_classes),
                                         dtype=np.float32)
                    batch_whs = np.zeros(
                        (self.batch_size, self.max_objects, 2), dtype=np.float32)
                    batch_regs = np.zeros(
                        (self.batch_size, self.max_objects, 2), dtype=np.float32)
                    batch_reg_masks = np.zeros(
                        (self.batch_size, self.max_objects), dtype=np.float32)
                    batch_indices = np.zeros(
                        (self.batch_size, self.max_objects), dtype=np.float32)

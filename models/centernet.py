import keras
import tensorflow as tf
from keras.models import *
from keras.layers import *
from keras.backend import *
from keras.optimizers import *
from keras.initializers import *
from keras.regularizers import *

from models.centernet_head import centernet_head
from models.resnet50 import ResNet50
from models.hrnet import HRNet
from models.shufflenet_v2 import ShuffleNetV2


def nms(heat, kernel=3):
    hmax = MaxPooling2D((kernel, kernel), strides=1, padding='same')(heat)
    heat = tf.where(tf.equal(hmax, heat), heat, tf.zeros_like(heat))
    return heat


def topk(hm, k=100):
    hm = nms(hm)
    b, h, w, c = tf.shape(hm)[0], tf.shape(hm)[1], tf.shape(hm)[2], tf.shape(hm)[3]
    hm = tf.reshape(hm, (b, -1))
    scores, indices = tf.math.top_k(hm, k=k, sorted=True)
    class_ids = indices % c
    xs = indices // c % w
    ys = indices // c // w
    indices = ys * w + xs
    return scores, indices, class_ids, xs, ys


def CenterNet(input_shape, num_class, backbone='resnet50', k=100, mode="train", num_stacks=2):
    output_size = input_shape[0] // 4
    image_input = Input(shape=input_shape)
    hm_input = Input(shape=(output_size, output_size, num_class))
    wh_input = Input(shape=(k, 2))
    reg_input = Input(shape=(k, 2))
    reg_mask_input = Input(shape=(k,))
    index_input = Input(shape=(k,))

    backbone_model = None

    if backbone == 'resnet50':
        backbone_model = ResNet50(image_input)
    elif backbone == 'hrnet':
        backbone_model = HRNet(image_input)
    elif backbone == 'shufflenet_v2':
        backbone_model = ShuffleNetV2(image_input)

    y1, y2, y3 = centernet_head(backbone_model, num_class)

    if mode == "train":
        loss_ = Lambda(loss, name='centernet_loss')(
            [y1, y2, y3, hm_input, wh_input, reg_input, reg_mask_input, index_input])
        model = Model(inputs=[image_input, hm_input, wh_input,
                                reg_input, reg_mask_input, index_input], outputs=[loss_])
        return model

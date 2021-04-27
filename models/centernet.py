import keras
import tensorflow as tf
from keras.models import *
from keras.layers import *
from keras.backend import *
from keras.optimizers import *
from keras.initializers import *
from keras.regularizers import *

from centernet_head import centernet_head


def nms(heat, kernel=3):
    hmax = MaxPooling2D((kernel, kernel), strides=1, padding='same')(heat)
    heat = tf.where(tf.equal(hmax, heat), heat, tf.zeros_like(heat))
    return heat


def topk(hm, k=100):
    hm = nms(hm)
    b, h, w, c = hm
    # b, h, w, c = tf.shape(hm)[0], tf.shape(hm)[1], tf.shape(hm)[2], tf.shape(hm)[3]
    hm = tf.reshape(hm, (b, -1))
    scores, indices = tf.math.top_k(hm, k=k, sorted=True)
    class_ids = indices % c
    xs = indices // c % w
    ys = indices // c // w
    indices = ys * w + xs
    return scores, indices, class_ids, xs, ys


def centernet(input_shape, num_classes, backbone='resnet50', max_objects=100, mode="train", num_stacks=2):
    assert backbone in ['resnet50', 'light_hrnet']
    output_size = input_shape[0] // 4
    image_input = Input(shape=input_shape)
    hm_input = Input(shape=(output_size, output_size, num_classes))
    wh_input = Input(shape=(max_objects, 2))
    reg_input = Input(shape=(max_objects, 2))
    reg_mask_input = Input(shape=(max_objects,))
    index_input = Input(shape=(max_objects,))

    if backbone == 'resnet50':
        resnet50 = ResNet50(image_input)
        y1, y2, y3 = centernet_head(resnet50, num_classes)

        if mode == "train":
            loss_ = Lambda(loss, name='centernet_loss')(
                [y1, y2, y3, hm_input, wh_input, reg_input, reg_mask_input, index_input])
            model = Model(inputs=[image_input, hm_input, wh_input,
                                  reg_input, reg_mask_input, index_input], outputs=[loss_])
            return model
        else:
            detections = Lambda(lambda x: decode(*x, max_objects=max_objects,
                                                 num_classes=num_classes))([y1, y2, y3])
            prediction_model = Model(inputs=image_input, outputs=detections)
            return prediction_model

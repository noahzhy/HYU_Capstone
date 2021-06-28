import keras
import tensorflow as tf

from keras import backend as K
from keras.layers import *
from keras.models import *
from keras.optimizers import *
from keras.activations import *
from tensorflow.keras.utils import plot_model


CEM_FILTER = 245
NUM_ANCHORS = 5*5
DEPTHWISE_CONV_KERNEL_SIZE = 5


def dwconv_bn_point(inputs, out_channel=256, kernel_size=3, strides=1):
    x = dwconv_bn(inputs, kernel_size=kernel_size, strides=strides,)
    x = Conv2D(out_channel, kernel_size=1, strides=1,
               padding="same", use_bias=True)(x)
    return x


def RPN(inputs):
    x = dwconv_bn_point(inputs, out_channel=256, kernel_size=5, strides=1)
    rpn_conf = Conv2D(2*NUM_ANCHORS, kernel_size=1, strides=1,
               padding="valid", use_bias=True)(x)
    rpn_pbbox = Conv2D(2*NUM_ANCHORS, kernel_size=1, strides=1,
                 padding="valid", use_bias=True)(x)
    return x, rpn_conf, rpn_pbbox


def CEM(stage2, stage3, stage4):
    c4_lat = Conv2D(CEM_FILTER, 1, strides=1, padding="same", use_bias=True)(stage2)
    c5_lat = Conv2D(CEM_FILTER, 1, strides=1, padding="same", use_bias=True)(stage3)
    c5_lat = K.resize_images(c5_lat, 2, 2, "channels_last", "bilinear")
    cglb_lat = K.reshape(stage4, [-1, 1, 1, CEM_FILTER])
    return c4_lat + c5_lat + cglb_lat


def SAM(rpn_in, cem_in):
    x = Conv2D(CEM_FILTER, 1, strides=1, padding="valid", use_bias=False)(rpn_in)
    x = BatchNormalization()(x)
    x = Activation("softmax")(x)
    x = Multiply()([x, cem_in])
    return x


def channel_split(x, num_splits=2):
    return tf.split(x, axis=-1, num_or_size_splits=num_splits)


def channel_shuffle(x, groups=2):
    n, h, w, c = x.get_shape().as_list()
    x = tf.reshape(x, [-1, h, w, groups, c // groups])
    x = tf.transpose(x, [0, 1, 2, 4, 3])
    x = tf.reshape(x, [-1, h, w, c])
    return x


def conv_bn_relu(inputs, out_channel, kernel_size=1, strides=1, relu="relu"):
    x = Conv2D(out_channel, kernel_size=kernel_size,
               strides=strides, padding="same", use_bias=False)(inputs)
    x = bn_relu(x, relu=relu)
    return x


def dwconv_bn(inputs, kernel_size=1, strides=1):
    x = DepthwiseConv2D(kernel_size=kernel_size, strides=strides,
                        padding="same", depth_multiplier=1, use_bias=False)(inputs)
    x = BatchNormalization()(x)
    return x


def conv_dwconv_conv(inputs, out_channel, strides=1, dwconv_ks=DEPTHWISE_CONV_KERNEL_SIZE):
    x = conv_bn_relu(inputs, out_channel, 1, 1, "relu")
    x = dwconv_bn(x, kernel_size=dwconv_ks, strides=strides)
    x = conv_bn_relu(x, out_channel, 1, 1, "relu")
    return x


def bn_relu(inputs, relu="relu"):
    x = BatchNormalization()(inputs)
    if relu == "relu":
        x = ReLU()(x)
    elif relu == "relu6":
        x = tf.nn.relu6(x)
    return x


def shufflenet_unit(inputs, out_channel, strides=1):
    half_channel = out_channel // 2

    if strides == 1:
        top, bottom = channel_split(inputs)
        top = conv_dwconv_conv(top, half_channel, strides)

    if strides == 2:
        top = conv_dwconv_conv(inputs, half_channel, strides)

        bottom = dwconv_bn(
            inputs, kernel_size=DEPTHWISE_CONV_KERNEL_SIZE, strides=strides)
        bottom = conv_bn_relu(bottom, half_channel, 1, 1, relu="relu")

    out = Concatenate()([top, bottom])
    out = channel_shuffle(out)
    return out


def stage(x, num_stages, out_channels):
    x = shufflenet_unit(x, out_channels, strides=2)
    for i in range(num_stages):
        x = shufflenet_unit(x, out_channels, strides=1)
    return x


def snet(inputs, out_channels: list, num_class=1000):
    x = conv_bn_relu(inputs, 24, kernel_size=3, strides=2)
    x = MaxPooling2D((3, 3), strides=2, padding='same')(x)

    s2 = stage(x, 3, out_channels[0])
    s3 = stage(s2, 7, out_channels[1])
    s4 = stage(s3, 3, out_channels[2])

    # if out_channels[3] > 0:
    #     x = conv_bn_relu(x, out_channels[3], relu="relu")

    cglb = GlobalAveragePooling2D()(s4)
    x = Dense(num_class, activation="softmax")(cglb)

    # model = Model(inputs=inputs, outputs=x)
    # return model
    return x, s3, s4, cglb


def snet_x(inputs, scale=0):
    # if scale == 0:
    #     out_channels = [60, 120, 240, 512]
    if scale == 1:
        out_channels = [132, 264, 528, 0]
    if scale == 2:
        out_channels = [248, 496, 992, 0]

    return snet(inputs, out_channels=out_channels)


def head(backbone):
    x, s3, s4, cglb = backbone
    cem = CEM(s3, s4, cglb)
    rpn, rpn_conf, rpn_pbbox = RPN(cem)
    sam = SAM(rpn, cem)
    return sam, rpn, rpn_conf, rpn_pbbox


def build_model():
    inputs = Input(shape=(320, 320, 3))
    bb = snet_x(inputs, scale=1)
    sam, rpn, rpn_conf, rpn_pbbox = head(bb)
    model = Model(inputs=inputs, outputs=[sam, rpn, rpn_conf, rpn_pbbox])
    model.summary()


if __name__ == '__main__':
    build_model()
    # plot_model(model, to_file='ShuffleNetV2.png',
    #            show_layer_names=True, show_shapes=True)

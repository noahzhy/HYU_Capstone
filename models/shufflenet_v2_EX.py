import keras

from keras.layers import *
from keras.models import *
from keras.optimizers import *
from keras.activations import *
from tensorflow.keras.utils import plot_model

import tensorflow as tf
import keras.backend as K
 

DEPTHWISE_CONV_KERNEL_SIZE = 5


def channel_split(x, num_splits=2):
    return tf.split(x, axis=-1, num_or_size_splits=num_splits)


def channel_shuffle(x, groups=2):
    n, h, w, c = x.get_shape().as_list()
    x = tf.reshape(x, [-1, h, w, groups, c // groups])
    x = tf.transpose(x, [0, 1, 2, 4, 3])
    x = tf.reshape(x, [-1, h, w, c])
    return x


def conv_bn_relu(inputs, out_channel, kernel_size=1, strides=1, relu="relu"):
    x = Conv2D(out_channel, kernel_size=kernel_size, strides=strides, padding="same", use_bias=False)(inputs)
    x = bn_relu(x, relu=relu)
    return x


def dwconv_bn(inputs, kernel_size=1, strides=1):
    x = DepthwiseConv2D(kernel_size=kernel_size, strides=strides, padding="same", use_bias=False)(inputs)
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

        bottom = dwconv_bn(inputs, kernel_size=DEPTHWISE_CONV_KERNEL_SIZE, strides=strides)
        bottom = conv_bn_relu(bottom, half_channel, 1, 1, relu="relu")

    out = Concatenate()([top, bottom])
    out = channel_shuffle(out)
    return out


def stage(x, num_stages, out_channels):
    x = shufflenet_unit(x, out_channels, strides=2)
    for i in range(num_stages):
        x = shufflenet_unit(x, out_channels, strides=1)
    return x


def shufflenet_v2(inputs, out_channels: list, num_class=1000):
    x = conv_bn_relu(inputs, 24, kernel_size=3, strides=2)
    x = MaxPooling2D((3, 3), strides=2, padding='same')(x)

    x = stage(x, 3, out_channels[0])
    x = stage(x, 7, out_channels[1])
    x = stage(x, 3, out_channels[2])

    x = conv_bn_relu(x, out_channels[3], relu="relu")

    x = GlobalAveragePooling2D()(x)
    x = Dense(num_class, activation="softmax")(x)

    model = Model(inputs=inputs, outputs=x)
    return model


def shufflenet_head(inputs, out_channel):
    split0, split1, split2 = tf.split(inputs, num_or_size_splits=3, axis=1)
    splited_list = [split0, split1, split2]

    for i in splited_list:
        print(i.shape())

    x0 = conv_dwconv_conv(top, out_channel, 1)

    out = concatenate()[x, x0]


def shufflenetV2_x(inputs, scale=1):
    if scale == 0.5:
        out_channels = [48, 96, 192, 1024]
    elif scale == 1:
        out_channels = [116, 232, 464, 1024]
    elif scale == 1.5:
        out_channels = [176, 352, 704, 1024]
    elif scale == 2:
        out_channels = [244, 488, 976, 2048]

    return shufflenet_v2(inputs, out_channels=out_channels)


if __name__ == '__main__':
    inputs = Input(shape=(224, 224, 3))
    model = shufflenetV2_x(inputs, scale=1)
    model.summary()
    # plot_model(model, to_file='ShuffleNetV2.png',
    #            show_layer_names=True, show_shapes=True)
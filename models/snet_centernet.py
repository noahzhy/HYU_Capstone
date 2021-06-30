import keras
import tensorflow as tf

from keras import backend as K
from keras.layers import *
from keras.models import *
from keras.optimizers import *
from keras.activations import *
from tensorflow.keras.utils import plot_model

from roi_utils import *


DEPTHWISE_CONV_KERNEL_SIZE = 5
RNI = tf.random_normal_initializer(stddev=0.01)


def channel_split(x, num_splits=2):
    return tf.split(x, axis=-1, num_or_size_splits=num_splits)


def channel_shuffle(x, groups=2):
    n, h, w, c = x.get_shape().as_list()
    x = tf.reshape(x, [-1, h, w, groups, c // groups])
    x = tf.transpose(x, [0, 1, 2, 4, 3])
    x = tf.reshape(x, [-1, h, w, c])
    return x


# def dwconv_bn_point(inputs, out_channel=256, kernel_size=3, strides=1):
#     x = dwconv_bn(inputs, kernel_size=kernel_size, strides=strides)
#     x = Conv2D(out_channel, kernel_size=1, strides=1,
#                padding="same", use_bias=True)(x)
#     return x


def conv_bn_relu(inputs, out_channel, kernel_size=1, strides=1, kernel_initializer="glorot_uniform", bn=True, relu="relu"):
    x = Conv2D(out_channel, kernel_size=kernel_size, kernel_initializer=kernel_initializer,
               strides=strides, padding="same", use_bias=False)(inputs)
    x = bn_relu(x, bn=bn, relu=relu)
    return x


def dwconv_bn(inputs, kernel_size=1, strides=1):
    x = DepthwiseConv2D(kernel_size=kernel_size, strides=strides,
                        padding="same", depth_multiplier=1, use_bias=False)(inputs)
    x = BatchNormalization()(x)
    return x


def deconv_bn_relu(inputs, out_channel, kernel_size=3, strides=2, relu="relu"):
    x = Conv2DTranspose(out_channel, kernel_size=kernel_size, strides=strides, padding="same", use_bias=False)(inputs)
    x = bn_relu(x, relu=relu)
    return x


def conv_dwconv_conv(inputs, out_channel, strides=1, dwconv_ks=DEPTHWISE_CONV_KERNEL_SIZE):
    x = conv_bn_relu(inputs, out_channel, 1, 1, bn=True, relu="relu")
    x = dwconv_bn(x, kernel_size=dwconv_ks, strides=strides)
    x = conv_bn_relu(x, out_channel, 1, 1, bn=True, relu="relu")
    return x


def bn_relu(x, bn=True, relu="relu"):
    if bn:
        x = BatchNormalization()(x)
    if relu == "relu":
        x = ReLU()(x)
    elif relu == "relu6":
        x = tf.nn.relu6(x)
    elif relu == "sigmoid":
        x = tf.nn.sigmoid(x)
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


def snet(inputs, out_channels: list, num_class=2, feature_channels=128):
    x = conv_bn_relu(inputs, 24, kernel_size=3, strides=2)
    out_4 = MaxPooling2D((3, 3), strides=2, padding='same')(x)

    out_8 = stage(out_4, 3, out_channels[0])
    out_16 = stage(out_8, 7, out_channels[1])
    out_32 = stage(out_16, 3, out_channels[2])

    deconv1 = deconv_bn_relu(out_32, feature_channels)
    out_16 = conv_bn_relu(out_16, feature_channels, 1)
    fuse1 = deconv1 + out_16

    deconv2 = deconv_bn_relu(fuse1, feature_channels)
    out_8 = conv_bn_relu(out_8, feature_channels, 1)
    fuse2 = out_8 + deconv2

    deconv3 = deconv_bn_relu(fuse2, feature_channels)
    out_4 = conv_bn_relu(out_4, feature_channels, 1)
    fuse3 = out_4 + deconv3

    features = SE_module(fuse3)
    features = SA_module(features)

    # detector
    center = conv_bn_relu(features, feature_channels, 3, 1, RNI, None, "relu")
    center = conv_bn_relu(center, num_class, 1, 1, RNI, None, "sigmoid")
    offset = conv_bn_relu(features, feature_channels, 3, 1, RNI, None, "relu")
    offset = conv_bn_relu(offset, 2, 1, 1, RNI, None, "sigmoid")
    size = conv_bn_relu(features, feature_channels, 3, 1, RNI, None, "relu")
    size = conv_bn_relu(size, 2, 1, 1, RNI, None, "relu")

    out = [center, offset, size]

    return Model(inputs=inputs, outputs=out)


def snet_x(inputs, scale=146):
    if scale == 49:
        out_channels = [60, 120, 240, 512]
    if scale == 146:
        out_channels = [132, 264, 528, 0]
    if scale == 535:
        out_channels = [248, 496, 992, 0]

    return snet(inputs, out_channels=out_channels)


def SA_module(inputs):
    shape = inputs.get_shape().as_list()
    x = Conv2D(shape[-1], kernel_size=1, strides=1)(inputs)
    x = BatchNormalization(fused=False)(x)
    x = Activation(activation="sigmoid")(x)
    out = Multiply()([inputs, x])
    return out


def SE_module(inputs, bottleneck=2):
    n_channel = inputs.get_shape().as_list()[-1]
    x = GlobalAveragePooling2D()(inputs)
    x = Dense(bottleneck, activation='relu')(x)
    x = Dense(n_channel, activation='sigmoid')(x)
    x = Multiply()([inputs, x])
    return x


def build_model():
    inputs = Input(shape=(320, 320, 3))
    model = snet_x(inputs, scale=146)
    model.summary()


if __name__ == '__main__':
    build_model()
    # plot_model(model, to_file='ShuffleNetV2.png',
    #            show_layer_names=True, show_shapes=True)

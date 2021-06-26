import keras
import tensorflow as tf

from keras import backend as K
from keras.layers import *
from keras.models import *
from keras.optimizers import *
from keras.activations import *
# from keras.utils import plot_model


def channel_split(x, num_splits=2):
    if num_splits == 2:
        return tf.split(x, axis=-1, num_or_size_splits=num_splits)
    else:
        raise ValueError('Error! num_splits should be 2')


def channel_shuffle(x):
    height, width, channels = x.shape.as_list()[1:]
    channels_per_split = channels // 2
    x = K.reshape(x, [-1, height, width, 2, channels_per_split])
    x = K.permute_dimensions(x, (0, 1, 2, 4, 3))
    x = K.reshape(x, [-1, height, width, channels])
    return x


def bn_relu(inputs, relu="relu"):
    x = BatchNormalization()(inputs)
    if relu == "relu":
        x = ReLU()(x)
    elif relu == "relu6":
        x = tf.nn.relu6(x)
    return x


def conv_bn_relu(inputs, out_channel, kernel_size=1, stride=1, relu="relu"):
    x = Conv2D(out_channel, kernel_size=kernel_size, strides=stride, padding="same", use_bias=False)(inputs)
    x = bn_relu(x, relu=relu)
    return x


def conv_dwconv_conv(inputs, out_channel, stride=1, dwconv_ks=3):
    x = conv_bn_relu(inputs, out_channel, 1, 1, "relu")
    x = DepthwiseConv2D(kernel_size=dwconv_ks, strides=stride, padding="same", use_bias=False)(x)
    x = bn_relu(x, relu=None)
    x = conv_bn_relu(x, out_channel, 1, 1, "relu")
    return x


def deconv_bn_relu(inputs, out_channel, kernel_size=4, stride=2, relu="relu"):
    x = Conv2DTranspose(out_channel, kernel_size=kernel_size, strides=stride, padding="same", use_bias=False)(inputs)
    x = bn_relu(x, relu=relu)
    return x


def shufflenet_unit(inputs, out_channel, stride=1):
    out_channel //= 2
    top, bottom = channel_split(inputs)
    top = conv_dwconv_conv(top, out_channel, stride)

    if stride == 2:
        bottom = DepthwiseConv2D(kernel_size=3, strides=stride, padding="same", use_bias=False)(inputs)
        bottom = bn_relu(bottom, None)
        bottom = conv_bn_relu(bottom, out_channel, 1, 1, relu="relu")

    out = Concatenate()([top, bottom])
    out = channel_shuffle(out)
    return out


def stage(x, num_stages, out_channels):
    x = shufflenet_unit(x, out_channels, stride=2)
    for i in range(num_stages):
        x = shufflenet_unit(x, out_channels, stride=1)
    return x


def shufflenetV2_centernet(inputs, out_channels: list, num_class=1, deconv_out=128):
    # x = Conv2D(24, (3, 3), strides=2, padding='same', use_bias=False)(inputs)
    # x = BatchNormalization()(x)
    # x = MaxPooling2D((3, 3), strides=2, padding='same')(x)

    out_0 = conv_bn_relu(inputs, 24, 3, 2, "relu")
    out_0 = conv_bn_relu(out_0, 24, 3, 2, "relu")

    out_1 = stage(out_0, 3, out_channels[0])
    out_2 = stage(out_1, 7, out_channels[1])
    out_3 = stage(out_2, 3, out_channels[2])

    deconv1 = deconv_bn_relu(out_3, deconv_out)
    out_2 = conv_bn_relu(out_2, deconv_out, 1)
    fuse1 = out_2 + deconv1

    deconv2 = deconv_bn_relu(fuse1, deconv_out)
    out_1 = conv_bn_relu(out_1, deconv_out, 1)
    fuse2 = out_1 + deconv2

    deconv3 = deconv_bn_relu(fuse2, deconv_out)
    out_0 = conv_bn_relu(out_0, deconv_out, 1)
    fuse3 = out_0 + deconv3

    class_id = conv_bn_relu(fuse3, 64, 3, 1, None)
    class_id = Conv2D(num_class, 1, 1, activation="sigmoid")(class_id)

    obj_size = conv_bn_relu(fuse3, 64, 3, 1, None)
    obj_size = Conv2D(2, 1, 1, activation="relu")(obj_size) 

    # return class_id, obj_size

    # x = Conv2D(out_channels[3], kernel_size=1, padding='same', strides=1)(out_3)
    # x = bn_relu(x, relu="relu")

    # x = GlobalAveragePooling2D()(x)
    # x = Dense(num_class, activation="softmax")(x)

    model = Model(inputs=inputs, outputs=[class_id, obj_size])
    return model


def shufflenetV2_x(inputs, scale=1):
    if scale == 0.5:
        out_channels = [48, 96, 192, 1024]
    elif scale == 1:
        out_channels = [116, 232, 464, 1024]
    elif scale == 1.5:
        out_channels = [176, 352, 704, 1024]
    elif scale == 2:
        out_channels = [244, 488, 976, 2048]

    return shufflenetV2_centernet(inputs, out_channels=out_channels)


if __name__ == '__main__':
    inputs = Input(shape=(224, 224, 3))
    model = shufflenetV2_x(inputs, scale=1)
    model.summary()
    # plot_model(model, to_file='ShuffleNetV2.png',
    #            show_layer_names=True, show_shapes=True)

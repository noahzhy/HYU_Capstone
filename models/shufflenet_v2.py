import keras
import tensorflow as tf
from keras import backend as K
from keras.layers import *
from keras.models import *
from keras.optimizers import *
from keras.activations import *
from keras.utils import plot_model


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


def ShuffleNetUnit(inputs, out_channels, stride=1):
    out_channels = out_channels // 2

    if stride == 1:
        residual, short_cut = channel_split(inputs)
        inputs = short_cut

    x = Conv2D(out_channels, (1, 1), use_bias=False)(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = DepthwiseConv2D((3, 3), strides=stride,
                        padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Conv2D(out_channels, (1, 1), use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    if stride == 1:
        ret = Concatenate(axis=-1)([x, residual])
    else:
        s = DepthwiseConv2D((3, 3), strides=stride,
                            padding='same', use_bias=False)(inputs)
        s = BatchNormalization()(s)
        s = Conv2D(out_channels, (1, 1), use_bias=False)(s)
        s = BatchNormalization()(s)
        s = Activation('relu')(s)
        ret = Concatenate(axis=-1)([x, s])

    ret = Lambda(channel_shuffle)(ret)
    return ret


def stage(x, num_stages, out_channels):
    x = ShuffleNetUnit(x, out_channels, stride=2)
    for i in range(num_stages):
        x = ShuffleNetUnit(x, out_channels, stride=1)
    return x


def ShuffleNetV2(inputs, out_channels: list, num_class=1000):
    x = Conv2D(24, (3, 3), strides=2, padding='same', use_bias=False)(inputs)
    x = BatchNormalization()(x)
    x = MaxPooling2D((3, 3), strides=2, padding='same')(x)

    x = stage(x, 3, out_channels[0])
    x = stage(x, 7, out_channels[1])
    x = stage(x, 3, out_channels[2])

    x = Conv2D(out_channels[3], kernel_size=1, padding='same', strides=1)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = GlobalAveragePooling2D()(x)
    x = Dense(num_class)(x)
    x = Activation('softmax')(x)

    model = Model(inputs=inputs, outputs=x)
    return model


def ShuffleNetV2_x(inputs, scale=1):
    if scale == 0.5:
        out_channels = [48, 96, 192, 1024]
    elif scale == 1:
        out_channels = [116, 232, 464, 1024]
    elif scale == 1.5:
        out_channels = [176, 352, 704, 1024]
    elif scale == 2:
        out_channels = [244, 488, 976, 2048]

    return ShuffleNetV2(inputs, out_channels=out_channels)


if __name__ == '__main__':
    inputs = Input(shape=(224, 224, 3))
    model = ShuffleNetV2_x(inputs, scale=1)
    model.summary()
    plot_model(model, to_file='ShuffleNetV2.png',
               show_layer_names=True, show_shapes=True)

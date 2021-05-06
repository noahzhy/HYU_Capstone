import keras
from keras.models import *
from keras.layers import *
from keras.backend import *
from keras.optimizers import *
from keras.initializers import *

from keras.regularizers import *


def centernet_head(x, num_classes, num_filters=256):
    x = Dropout(rate=0.5)(x)

    for i in range(3):
        x = Conv2DTranspose(
            num_filters // pow(2, i),
            (4, 4), strides=2, use_bias=False, padding='same',
            kernel_initializer='he_normal',
            kernel_regularizer=l2(5e-4))(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

    # heatmap header
    y1 = DepthwiseConv2D(64, 5, padding='same', use_bias=False,
                kernel_initializer='he_normal', kernel_regularizer=l2(5e-4))(x)
    y1 = BatchNormalization()(y1)
    y1 = Activation('relu')(y1)
    y1 = Conv2D(num_classes, 1, kernel_initializer='he_normal',
                kernel_regularizer=l2(5e-4), activation='sigmoid')(y1)

    # size header
    y2 = DepthwiseConv2D(64, 5, padding='same', use_bias=False,
                kernel_initializer='he_normal', kernel_regularizer=l2(5e-4))(x)
    y2 = BatchNormalization()(y2)
    y2 = Activation('relu')(y2)
    y2 = Conv2D(2, 1, kernel_initializer='he_normal',
                kernel_regularizer=l2(5e-4))(y2)

    # reg(offset) header
    y3 = DepthwiseConv2D(64, 5, padding='same', use_bias=False,
                kernel_initializer='he_normal', kernel_regularizer=l2(5e-4))(x)
    y3 = BatchNormalization()(y3)
    y3 = Activation('relu')(y3)
    y3 = Conv2D(2, 1, kernel_initializer='he_normal',
                kernel_regularizer=l2(5e-4))(y3)

    return y1, y2, y3

from __future__ import absolute_import
from __future__ import print_function

import os
import numpy as np

from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Activation, UpSampling2D, BatchNormalization, add
from keras.layers.core import Flatten, Reshape
from keras.layers.convolutional import Convolution2D
from keras.models import Model
from keras.utils import np_utils
from keras.applications import imagenet_utils
from keras.regularizers import l2
import keras.backend as K

from .BilinearUpSampling import *

def _shortcut(input, residual):
    """Adds a shortcut between input and residual block and merges them with "sum"
    """
    # Expand channels of shortcut to match residual.
    # Stride appropriately to match residual (width, height)
    # Should be int if network architecture is correctly configured.
    input_shape = K.int_shape(input)
    residual_shape = K.int_shape(residual)
    stride_width = int(round(input_shape[1] / residual_shape[1]))
    stride_height = int(round(input_shape[2] / residual_shape[2]))
    equal_channels = input_shape[3] == residual_shape[3]

    shortcut = input
    # 1 X 1 conv if shape is different. Else identity.
    if stride_width > 1 or stride_height > 1 or not equal_channels:
        shortcut = Conv2D(filters=residual_shape[3],
                          kernel_size=(1, 1),
                          strides=(stride_width, stride_height),
                          padding="valid",
                          kernel_initializer="he_normal",
                          kernel_regularizer=l2(0.0001))(input)

    return add([shortcut, residual])

def encoder_block_by_numb(input_tensor, m, n, numb, FirstLayer=False):
    if FirstLayer:
        x = Conv2D(filters=n, kernel_size=(3, 3), strides=(2, 2), padding="same")(input_tensor)
    else:
        x = BatchNormalization()(input_tensor)
        x = Activation('relu')(x)
        x = Conv2D(filters=n, kernel_size=(3, 3), strides=(2, 2), padding="same")(x)

    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(filters=n, kernel_size=(3, 3), padding="same")(x)

    x = _shortcut(input_tensor, x)

    for i in range(1, numb):
        prev_out = x
        x = BatchNormalization()(prev_out)
        x = Activation('relu')(x)
        x = Conv2D(filters=n, kernel_size=(3, 3), padding="same")(x)

        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv2D(filters=n, kernel_size=(3, 3), padding="same")(x)

        x = _shortcut(prev_out, x)

    return x


def encoder_block_by_numb_bottlenet(input_tensor, m, n, numb, FirstLayer=False):
    if FirstLayer:
        x = Conv2D(filters=n, kernel_size=(3, 3), strides=(2, 2), padding="same")(input_tensor)
    else:
        x = BatchNormalization()(input_tensor)
        x = Activation('relu')(x)
        x = Conv2D(filters=m, kernel_size=(3, 3), strides=(2, 2), padding="same")(x)

    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(filters=m, kernel_size=(3, 3), padding="same")(x)

    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(filters=n, kernel_size=(1, 1), padding="same")(x)

    x = _shortcut(input_tensor, x)

    for i in range(1, numb):
        prev_out = x
        x = BatchNormalization()(prev_out)
        x = Activation('relu')(x)
        x = Conv2D(filters=m, kernel_size=(1, 1), padding="same")(x)

        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv2D(filters=m, kernel_size=(3, 3), padding="same")(x)

        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv2D(filters=n, kernel_size=(1, 1), padding="same")(x)

        x = _shortcut(prev_out, x)

    return x

# def encoder_block_mod(input_tensor, m, n):
#     x = BatchNormalization()(input_tensor)
#     x = Activation('relu')(x)
#     x = Conv2D(filters=n, kernel_size=(3, 3), strides=(2, 2), padding="same")(x)
#
#     x = BatchNormalization()(x)
#     x = Activation('relu')(x)
#     x = Conv2D(filters=n, kernel_size=(3, 3), padding="same")(x)
#
#     added_1 = _shortcut(input_tensor, x)
#
#     x = BatchNormalization()(added_1)
#     x = Activation('relu')(x)
#     x = Conv2D(filters=n, kernel_size=(3, 3), padding="same")(x)
#
#     x = BatchNormalization()(x)
#     x = Activation('relu')(x)
#     x = Conv2D(filters=n, kernel_size=(3, 3), padding="same")(x)
#
#     added_2 = _shortcut(added_1, x)
#
#     return added_2

def decoder_block_mod(input_tensor, m, n):
    x = BatchNormalization()(input_tensor)
    x = Activation('relu')(x)
    x = Conv2D(filters=int(m/4), kernel_size=(1, 1))(x)

    x = UpSampling2D((2, 2))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(filters=int(m/4), kernel_size=(3, 3), padding='same')(x)

    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(filters=n, kernel_size=(1, 1))(x)

    return x

# def encoder_block(input_tensor, m, n):
#     x = Conv2D(filters=n, kernel_size=(3, 3), strides=(2, 2), padding="same")(input_tensor)
#     x = BatchNormalization()(x)
#     x = Activation('relu')(x)
#
#     x = Conv2D(filters=n, kernel_size=(3, 3), padding="same")(x)
#     x = BatchNormalization()(x)
#     # x = Activation('relu')(x)
#
#     added_1 = _shortcut(input_tensor, x)
#     added_1_relu = Activation('relu')(added_1)
#
#     x = Conv2D(filters=n, kernel_size=(3, 3), padding="same")(added_1_relu)
#     x = BatchNormalization()(x)
#     x = Activation('relu')(x)
#
#     x = Conv2D(filters=n, kernel_size=(3, 3), padding="same")(x)
#     x = BatchNormalization()(x)
#     # x = Activation('relu')(x)
#
#     added_2 = _shortcut(added_1_relu, x)
#     added_2_relu = Activation('relu')(added_2)
#
#     return added_2_relu
#
# def decoder_block(input_tensor, m, n):
#     x = Conv2D(filters=int(m/4), kernel_size=(1, 1))(input_tensor)
#     x = BatchNormalization()(x)
#     x = Activation('relu')(x)
#
#     x = UpSampling2D((2, 2))(x)
#     x = Conv2D(filters=int(m/4), kernel_size=(1, 1))(x)
#     x = BatchNormalization()(x)
#     x = Activation('relu')(x)
#
#     x = Conv2D(filters=n, kernel_size=(1, 1))(x)
#     x = BatchNormalization()(x)
#     x = Activation('relu')(x)
#
#     return x

def LinkNet_Res50(input_shape=(256, 256, 3), classes=1):
    inputs = Input(shape=input_shape)

    x = BatchNormalization()(inputs)
    x = Activation('relu')(x)
    x = Conv2D(filters=64, kernel_size=(7, 7), strides=(2, 2))(x)


    x = MaxPooling2D((3, 3), strides=(2, 2), padding="same")(x)

    encoder_1 = encoder_block_by_numb_bottlenet(input_tensor=x, m=64, n=256, numb=3, FirstLayer=True)

    encoder_2 = encoder_block_by_numb_bottlenet(input_tensor=encoder_1, m=128, n=512, numb=4)

    encoder_3 = encoder_block_by_numb_bottlenet(input_tensor=encoder_2, m=256, n=1024, numb=6)

    encoder_4 = encoder_block_by_numb_bottlenet(input_tensor=encoder_3, m=512, n=2048, numb=3)

    # central = encoder_block_by_numb(input_tensor=encoder_4, m=512, n=1024)
    #
    # decoder_5 = decoder_block(input_tensor=central, m=1024, n=512)
    #
    # decoder_4_in = add([decoder_5, encoder_4])
    # decoder_4_in = Activation('relu')(decoder_4_in)

    decoder_4 = decoder_block_mod(input_tensor=encoder_4, m=2048, n=1024)

    decoder_3_in = add([encoder_3, decoder_4])
    decoder_3_in = Activation('relu')(decoder_3_in)

    decoder_3 = decoder_block_mod(input_tensor=decoder_3_in, m=1024, n=512)

    decoder_2_in = add([encoder_2, decoder_3])
    decoder_2_in = Activation('relu')(decoder_2_in)

    decoder_2 = decoder_block_mod(input_tensor=decoder_2_in, m=512, n=256)

    decoder_1_in = add([encoder_1, decoder_2])
    decoder_1_in = Activation('relu')(decoder_1_in)

    decoder_1 = decoder_block_mod(input_tensor=decoder_1_in, m=256, n=64)

    x = UpSampling2D((2, 2))(decoder_1)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(filters=32, kernel_size=(3, 3), padding="same")(x)

    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(filters=32, kernel_size=(3, 3), padding="same")(x)

    x = UpSampling2D((2, 2))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(filters=classes, kernel_size=(2, 2), padding="same")(x)
    # classify = Conv2D(classes, (1, 1), activation='sigmoid')(x)

    model = Model(inputs=inputs, outputs=x)

    return model

def LinkNet_Res34(input_shape=(256, 256, 3), classes=1):
    inputs = Input(shape=input_shape)

    x = BatchNormalization()(inputs)
    x = Activation('relu')(x)
    x = Conv2D(filters=64, kernel_size=(7, 7), strides=(2, 2))(x)

    x = MaxPooling2D((3, 3), strides=(2, 2), padding="same")(x)

    encoder_1 = encoder_block_by_numb(input_tensor=x, m=64, n=64, numb=3, FirstLayer=True)

    encoder_2 = encoder_block_by_numb(input_tensor=encoder_1, m=64, n=128, numb=4)

    encoder_3 = encoder_block_by_numb(input_tensor=encoder_2, m=128, n=256, numb=6)

    encoder_4 = encoder_block_by_numb(input_tensor=encoder_3, m=256, n=512, numb=3)

    # central = encoder_block_by_numb(input_tensor=encoder_4, m=512, n=1024)
    #
    # decoder_5 = decoder_block(input_tensor=central, m=1024, n=512)
    #
    # decoder_4_in = add([decoder_5, encoder_4])
    # decoder_4_in = Activation('relu')(decoder_4_in)

    decoder_4 = decoder_block_mod(input_tensor=encoder_4, m=512, n=256)

    decoder_3_in = add([decoder_4, encoder_3])
    decoder_3_in = Activation('relu')(decoder_3_in)

    decoder_3 = decoder_block_mod(input_tensor=decoder_3_in, m=256, n=128)

    decoder_2_in = add([decoder_3, encoder_2])
    decoder_2_in = Activation('relu')(decoder_2_in)

    decoder_2 = decoder_block_mod(input_tensor=decoder_2_in, m=128, n=64)

    decoder_1_in = add([decoder_2, encoder_1])
    decoder_1_in = Activation('relu')(decoder_1_in)

    decoder_1 = decoder_block_mod(input_tensor=decoder_1_in, m=64, n=64)

    x = UpSampling2D((2, 2))(decoder_1)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(filters=32, kernel_size=(3, 3), padding="same")(x)

    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(filters=32, kernel_size=(3, 3), padding="same")(x)

    x = UpSampling2D((2, 2))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(filters=classes, kernel_size=(2, 2), padding="same")(x)
    # classify = Conv2D(classes, (1, 1), activation='sigmoid')(x)

    model = Model(inputs=inputs, outputs=x)

    return model


# def LinkNet_mod(input_shape=(256, 256, 3), classes=1):
#     inputs = Input(shape=input_shape)
#
#     x = BatchNormalization()(inputs)
#     x = Activation('relu')(x)
#     x = Conv2D(filters=64, kernel_size=(7, 7), strides=(2, 2))(x)
#
#     x = MaxPooling2D((3, 3), strides=(2, 2), padding="same")(x)
#
#     encoder_1 = encoder_block_mod(input_tensor=x, m=64, n=64)
#
#     encoder_2 = encoder_block_mod(input_tensor=encoder_1, m=64, n=128)
#
#     encoder_3 = encoder_block_mod(input_tensor=encoder_2, m=128, n=256)
#
#     encoder_4 = encoder_block_mod(input_tensor=encoder_3, m=256, n=512)
#
#     central = encoder_block_mod(input_tensor=encoder_4, m=512, n=1024)
#
#     decoder_5 = decoder_block_mod(input_tensor=central, m=1024, n=512)
#
#     decoder_4_in = add([decoder_5, encoder_4])
#     decoder_4_in = Activation('relu')(decoder_4_in)
#
#     decoder_4 = decoder_block_mod(input_tensor=decoder_4_in, m=512, n=256)
#
#     decoder_3_in = add([decoder_4, encoder_3])
#     decoder_3_in = Activation('relu')(decoder_3_in)
#
#     decoder_3 = decoder_block_mod(input_tensor=decoder_3_in, m=256, n=128)
#
#     decoder_2_in = add([decoder_3, encoder_2])
#     decoder_2_in = Activation('relu')(decoder_2_in)
#
#     decoder_2 = decoder_block_mod(input_tensor=decoder_2_in, m=128, n=64)
#
#     decoder_1_in = add([decoder_2, encoder_1])
#     decoder_1_in = Activation('relu')(decoder_1_in)
#
#     decoder_1 = decoder_block_mod(input_tensor=decoder_1_in, m=64, n=64)
#
#     x = UpSampling2D((2, 2))(decoder_1)
#     x = BatchNormalization()(x)
#     x = Activation('relu')(x)
#     x = Conv2D(filters=32, kernel_size=(3, 3), padding="same")(x)
#
#     x = BatchNormalization()(x)
#     x = Activation('relu')(x)
#     x = Conv2D(filters=32, kernel_size=(3, 3), padding="same")(x)
#
#     x = UpSampling2D((2, 2))(x)
#     x = BatchNormalization()(x)
#     x = Activation('relu')(x)
#     x = Conv2D(filters=classes, kernel_size=(2, 2), padding="same")(x)
#     # classify = Conv2D(classes, (1, 1), activation='sigmoid')(x)
#
#     model = Model(inputs=inputs, outputs=x)
#
#     return model

def LinkNet(input_shape=(256, 256, 3), classes=1):
    inputs = Input(shape=input_shape)

    x = BatchNormalization()(inputs)
    x = Activation('relu')(x)
    x = Conv2D(filters=64, kernel_size=(7, 7), strides=(2, 2))(x)

    x = MaxPooling2D((3, 3), strides=(2, 2), padding="same")(x)

    encoder_1 = encoder_block_by_numb(input_tensor=x, m=64, n=64, numb=2, FirstLayer=True)

    encoder_2 = encoder_block_by_numb(input_tensor=encoder_1, m=64, n=128, numb=2)

    encoder_3 = encoder_block_by_numb(input_tensor=encoder_2, m=128, n=256, numb=2)

    encoder_4 = encoder_block_by_numb(input_tensor=encoder_3, m=256, n=512, numb=2)

    # central = encoder_block_by_numb(input_tensor=encoder_3, m=512, n=1024, numb=2)
    #
    # decoder_5 = decoder_block_mod(input_tensor=central, m=1024, n=512)
    #
    # decoder_4_in = add([decoder_5, encoder_4])
    # decoder_4_in = Activation('relu')(decoder_4_in)

    decoder_4 = decoder_block_mod(input_tensor=encoder_4, m=512, n=256)

    decoder_3_in = add([decoder_4, encoder_3])
    decoder_3_in = Activation('relu')(decoder_3_in)

    decoder_3 = decoder_block_mod(input_tensor=decoder_3_in, m=256, n=128)

    decoder_2_in = add([decoder_3, encoder_2])
    decoder_2_in = Activation('relu')(decoder_2_in)

    decoder_2 = decoder_block_mod(input_tensor=decoder_2_in, m=128, n=64)

    decoder_1_in = add([decoder_2, encoder_1])
    decoder_1_in = Activation('relu')(decoder_1_in)

    decoder_1 = decoder_block_mod(input_tensor=decoder_1_in, m=64, n=64)

    x = UpSampling2D((2, 2))(decoder_1)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(filters=32, kernel_size=(3, 3), padding="same")(x)

    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(filters=32, kernel_size=(3, 3), padding="same")(x)

    x = UpSampling2D((2, 2))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(filters=classes, kernel_size=(2, 2), padding="same")(x)
    # classify = Conv2D(classes, (1, 1), activation='sigmoid')(x)

    model = Model(inputs=inputs, outputs=x)

    return model
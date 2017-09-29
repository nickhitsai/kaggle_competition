from __future__ import absolute_import

from .enet_unpooling import model as enet_unpooling
from .enet_naive_upsampling import model as enet_naive_upsampling


def get_enet_unpooling(input_shape=(360, 480, 3), classes=12):
    return enet_unpooling.get_model(input_shape=input_shape, classes=classes)


def get_enet_naive_upsampling(input_shape=(360, 480, 3), classes=12):
    return enet_naive_upsampling.get_model(input_shape=input_shape, classes=classes)

import cv2, argparse, threading, sys, PIL
import numpy as np
import pandas as pd
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard, CSVLogger, LearningRateScheduler
from keras import losses, optimizers
from keras.layers import Conv2D, pooling
from keras.models import Model
from keras.utils import plot_model
import keras.backend as K
import tensorflow as tf

from sklearn.model_selection import train_test_split

from model.u_net import get_unet, get_unet_128, get_unet_256, get_unet_512, get_unet_1024, dice_loss, bce_dice_loss, get_init_unet, get_modify_unet, get_modify_unet_V2
from model.VGG16 import get_vgg16

from model.segnet import SegNet, SegNet_FULL
from model.enet import get_enet_unpooling, get_enet_naive_upsampling
from model.densenet_fc import DenseNetFCN
from model.linknet import LinkNet, LinkNet_Res34, LinkNet_Res50

import utils

class threadsafe_iter:
    """Takes an iterator/generator and makes it thread-safe by
    serializing call to the `next` method of given iterator/generator.
    """
    def __init__(self, it):
        self.it = it
        self.lock = threading.Lock()

    def __iter__(self):
        return self

    def __next__(self):
        with self.lock:
            return next(self.it)


def threadsafe_generator(f):
    """A decorator that takes a generator function and makes it thread-safe.
    """
    def g(*a, **kw):
        return threadsafe_iter(f(*a, **kw))
    return g


class Batch_Generator:
    def __init__(self, input_path, idx, batch_size, img_tuple, aug=False, hue=False,
                                                                          sat=False,
                                                                          val=False,
                                                                          rotate=False,
                                                                          gray=False,
                                                                          contrast=False,
                                                                          brightness=False,
                                                                          shear=False,
                                                                          shift=False,
                                                                          scale=False,
                                                                          lowq=False):
        super(Batch_Generator, self).__init__()
        self.input_path = input_path
        self.idx = idx
        self.batch_size = batch_size
        self.img_w, self.img_h, self.img_d = img_tuple
        self.aug = aug
        self.lock = threading.Lock()
        self.counter = 0

        self.hue = hue
        self.sat = sat
        self.val = val
        self.rotate = rotate
        self.gray = gray
        self.contrast = contrast
        self.brightness = brightness
        self.shear = shear
        self.shift = shift
        self.scale = scale
        self.lowq = lowq

    def __next__(self):
        start = self.counter * self.batch_size
        with self.lock:
            self.counter += 1
            if self.counter > len(self.idx)//self.batch_size - 1:
                self.counter = 0

        x_batch = []
        y_batch = []
        end = min(start + self.batch_size, len(self.idx))
        ids_train_batch = self.idx[start:end]
        for id in ids_train_batch.values:
            # img = cv2.imread(self.input_path + 'train/{}.jpg'.format(id))
            # img = cv2.resize(img, (self.img_size, self.img_size))
            if self.lowq:
                img = PIL.Image.open(self.input_path + 'train/{}.jpg'.format(id))
            else:
                img = PIL.Image.open(self.input_path + 'train_hq/{}.jpg'.format(id))

            img = np.array(img)
            img = cv2.resize(img.astype(np.float32), (self.img_h, self.img_w), interpolation=cv2.INTER_LINEAR)
            # mask = cv2.imread(self.input_path + 'train_masks/{}_mask.png'.format(id), cv2.IMREAD_GRAYSCALE)
            # mask = cv2.resize(mask, (self.img_size, self.img_size))
            mask = PIL.Image.open(self.input_path + 'train_masks/{}_mask.gif'.format(id))
            mask = np.array(mask)
            mask = cv2.resize(mask.astype(np.float32), (self.img_h, self.img_w), interpolation=cv2.INTER_LINEAR)
            mask = (mask>0.5).astype(np.float32)

            if self.aug:
                # rotate 45 > 15
                # shift 0.0625 > 0.1
                # scale 0.1 > 0.15
                # scale 0.1 > 0.05
                if self.rotate:
                    img, mask = randomShiftScaleRotate(img, mask,
                                                       shift_limit=(-0.0625, 0.0625),
                                                       scale_limit=(-0.1, 0.1),
                                                       rotate_limit=(-15, 15))
                if self.shift:
                    img, mask = randomShiftScaleRotate(img, mask,
                                                       shift_limit=(-0.1, 0.1),
                                                       scale_limit=(-0.1, 0.1),
                                                       rotate_limit=(-0, 0))
                if self.scale:
                    img, mask = randomShiftScaleRotate(img, mask,
                                                       shift_limit=(-0.0625, 0.0625),
                                                       scale_limit=(-0.05, 0.05),
                                                       rotate_limit=(-0, 0))
                else:
                    img, mask = randomShiftScaleRotate(img, mask,
                                                       shift_limit=(-0.0625, 0.0625),
                                                       scale_limit=(-0.1, 0.1),
                                                       rotate_limit=(-0, 0))

                img, mask = randomHorizontalFlip(img, mask)

                # img, mask = utils.random_augmentation(img, mask)
                if self.shear:
                    img, mask = random_shear(img, mask, intensity_range=(-0.3, 0.3), u=0.2)

                if self.contrast:
                    img = utils.random_contrast(img)

                if self.brightness:
                    img = utils.random_brightness(img)

                if self.gray:
                    img = randomGray(img, u=0.2)

                if self.hue:
                    img = randomHueSaturationValue(img,
                                   hue_shift_limit=(-50, 50),
                                   sat_shift_limit=(-0, 0),
                                   val_shift_limit=(-0, 0))

                if self.sat:
                    img = randomHueSaturationValue(img,
                                   hue_shift_limit=(-0, 0),
                                   sat_shift_limit=(-2, 2),
                                   val_shift_limit=(-0, 0))

                if self.val:
                    img = randomHueSaturationValue(img,
                                   hue_shift_limit=(-0, 0),
                                   sat_shift_limit=(-0, 0),
                                   val_shift_limit=(-5, 5))

                # img = randomHueSaturationValue(img,
                #                hue_shift_limit=(-0, 0),
                #             #    sat_shift_limit=(-0, 0),
                #             #    hue_shift_limit=(-50, 50),
                #                sat_shift_limit=(-5, 5),
                #             #    val_shift_limit=(-15, 15))
                #                val_shift_limit=(-0, 0))

            mask = np.expand_dims(mask, axis=2)
            x_batch.append(img)
            y_batch.append(mask)
        x_batch = np.array(x_batch, np.float32) / 255
        y_batch = np.array(y_batch, np.float32)
        return x_batch, y_batch
    def __iter__(self):
        return self

def unet_lr_scheduler(epoch_index):
    lr = 1e-02

    if epoch_index < 3:    # 0-2
        lr = 1e-03
    elif epoch_index < 11:  # 3-10
        lr = 3e-04
    elif epoch_index < 16:  # 10-15
        lr = 1e-04
    elif epoch_index < 51 and epoch_index > 45: # 46-50
        lr = 3e-04
    elif epoch_index < 16 and epoch_index > 14: # 21-30
        lr = 1e-04
    else:                                       # 31-35
        lr = 3e-05

    return lr

def linknet_lr_scheduler(epoch_index):
    lr = 1e-02

    if epoch_index < 39:                            # 1-38
        lr = 1e-02
    elif epoch_index < 49 and epoch_index > 38:       # 39-48
        lr = 3e-03
    elif epoch_index < 56 and epoch_index > 48:  # 49-55
        lr = 1e-03
    elif epoch_index < 61 and epoch_index > 54: # 55-60
        lr = 3e-04
    elif epoch_index < 16 and epoch_index > 14: # 21-30
        lr = 1e-04
    else:                                       # 31-35
        lr = 3e-05

    return lr

def enet_lr_scheduler(epoch_index):
    lr = 1e-02

    if epoch_index < 39:                            # 1-38
        lr = 1e-02
    elif epoch_index < 42:       # 39-48
        lr = 3e-03
    elif epoch_index < 45:  # 49-55
        lr = 1e-03
    elif epoch_index < 61: # 55-60
        lr = 3e-04
    elif epoch_index < 16 and epoch_index > 14: # 21-30
        lr = 1e-04
    else:                                       # 31-35
        lr = 3e-05

    return lr

def _to_tensor(x, dtype):
    """Convert the input `x` to a tensor of type `dtype`.

    # Arguments
        x: An object to be converted (numpy array, list, tensors).
        dtype: The destination type.

    # Returns
        A tensor.
    """
    x = tf.convert_to_tensor(x)
    if x.dtype != dtype:
        x = tf.cast(x, dtype)
    return x

# def custom_BCE_loss(y_true, y_pred, weights):
#     epsilon = _to_tensor(10e-8, y_pred.dtype.base_dtype)
#     y_pred = tf.clip_by_value(y_pred, epsilon, 1 - epsilon)
#     y_pred = tf.log(y_pred / (1 - y_pred))
#
#     loss = K.sum(weights*K.clip(y_pred, min_value=0, max_value=256) - weights * y_pred * y_true + weights * K.log(1 + K.exp(-K.abs(y_pred)))) / K.sum(weights)
#     return loss

def weighted_bce_loss(y_true, y_pred, weight):
    # avoiding overflow
    epsilon = 1e-7
    y_pred = K.clip(y_pred, epsilon, 1. - epsilon)
    logit_y_pred = K.log(y_pred / (1. - y_pred))

    # https://www.tensorflow.org/api_docs/python/tf/nn/weighted_cross_entropy_with_logits
    loss = (1. - y_true) * logit_y_pred + (1. + (weight - 1.) * y_true) * \
                                          (K.log(1. + K.exp(-K.abs(logit_y_pred))) + K.maximum(-logit_y_pred, 0.))
    return K.sum(loss) / K.sum(weight)

# def bce_dice_loss(y_true, y_pred):
#     return binary_crossentropy(y_true, y_pred) + (1 - dice_loss(y_true, y_pred))

# def weighted_dice_loss(y_true, y_pred, weights):
#     smooth = 1.
#     y_true_f = K.flatten(y_true)
#     y_pred_f = K.flatten(y_pred)
#     intersection = y_true_f * y_pred_f
#     w2 = K.flatten(weights*weights)
#     score = (2. * K.sum(w2 * intersection) + smooth) / (K.sum(w2 * y_true_f) + K.sum(w2 * y_pred_f) + smooth)
#     return 1 - score

def weighted_dice_loss(y_true, y_pred, weight):
    smooth = 1.
    w, m1, m2 = weight * weight, y_true, y_pred
    intersection = (m1 * m2)
    score = (2. * K.sum(w * intersection) + smooth) / (K.sum(w * m1) + K.sum(w * m2) + smooth)
    return 1 - score

def weighted_loss_none(y_true, y_pred, is_weight=True):
    weights = K.ones_like(x=y_true)

    l = custom_BCE_loss(y_true, y_pred, weights) + weighted_dice_loss(y_true, y_pred, weights)
    return l

def weighted_loss(y_true, y_pred, is_weight=True):
    y_true = K.cast(y_true, 'float32')
    y_pred = K.cast(y_pred, 'float32')
    # if we want to get same size of output, kernel size must be odd number
    if K.int_shape(y_pred)[1] == 128:
        kernel_size = 11
    elif K.int_shape(y_pred)[1] == 256:
        kernel_size = 21
    elif K.int_shape(y_pred)[1] == 512:
        kernel_size = 21
    elif K.int_shape(y_pred)[1] == 1024:
        kernel_size = 41
    else:
        raise ValueError('Unexpected image size')

    weights = K.ones_like(x=y_true)
    pooled = K.pool2d(x=y_true, pool_size=(kernel_size, kernel_size), strides=(1, 1), padding='same', pool_mode='avg')

    # ge = K.greater_equal(x=pooled, y=0.01)
    # ge = _to_tensor(x=ge, dtype=tf.float32)
    #
    # le = K.less_equal(x=pooled, y=0.99)
    # le = _to_tensor(x=le, dtype=tf.float32)
    #
    # ind = ge * le
    # ind_weight = pooled*ind
    #
    #
    # ge = K.greater_equal(x=pooled, y=0.2)
    # ge = _to_tensor(x=ge, dtype=tf.float32)
    #
    # le = K.less_equal(x=pooled, y=0.8)
    # le = _to_tensor(x=le, dtype=tf.float32)
    #
    # ind = ge * le
    # ind_weight_2 = pooled*ind

    ind_weight = K.cast(K.greater(pooled, 0.01), 'float32') * K.cast(K.less(pooled, 0.99), 'float32')
    # ind_weight_2 = K.cast(K.greater(pooled, 0.2), 'float32') * K.cast(K.less(pooled, 0.8), 'float32')

    if is_weight:
        w0 = K.sum(weights)
        # weights = weights + ind_weight*2 + ind_weight_2*2
        weights = weights + ind_weight*2
        w1 = K.sum(weights)
        weights = weights/w1*w0

    # weights = K.ones(shape=(6, 1024, 1024, 1))
    # l = custom_BCE_loss(y_true, y_pred, weights)
    # l = weighted_dice_loss(y_true, y_pred, weights)
    l = weighted_bce_loss(y_true, y_pred, weights) + weighted_dice_loss(y_true, y_pred, weights)
    return l


def weighted_loss_256(y_true, y_pred, is_weight=True):
    weights = K.ones_like(x=y_true)
    pooled = K.pool2d(x=y_true, pool_size=(11, 11), strides=(1, 1), padding='same', pool_mode='avg')

    # ge = K.greater_equal(x=pooled, y=0.01)
    # ge = _to_tensor(x=ge, dtype=tf.float32)
    #
    # le = K.less_equal(x=pooled, y=0.99)
    # le = _to_tensor(x=le, dtype=tf.float32)
    #
    # ind = ge * le
    # ind_weight = pooled*ind
    #
    #
    # ge = K.greater_equal(x=pooled, y=0.2)
    # ge = _to_tensor(x=ge, dtype=tf.float32)
    #
    # le = K.less_equal(x=pooled, y=0.8)
    # le = _to_tensor(x=le, dtype=tf.float32)
    #
    # ind = ge * le
    # ind_weight_2 = pooled*ind

    ind_weight = K.cast(K.greater(pooled, 0.01), 'float32') * K.cast(K.less(pooled, 0.99), 'float32')
    # ind_weight_2 = K.cast(K.greater(pooled, 0.2), 'float32') * K.cast(K.less(pooled, 0.8), 'float32')

    if is_weight:
        w0 = K.sum(weights)
        # weights = weights + ind_weight*2 + ind_weight_2*2
        weights = weights + ind_weight*2
        w1 = K.sum(weights)
        weights = weights/w1*w0

    # weights = K.ones(shape=(6, 1024, 1024, 1))
    # l = custom_BCE_loss(y_true, y_pred, weights)
    # l = weighted_dice_loss(y_true, y_pred, weights)
    l = custom_BCE_loss(y_true, y_pred, weights) + weighted_dice_loss(y_true, y_pred, weights)
    return l

def randomGray(image, u=0.5):
    if np.random.random() < u:
        grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = np.repeat(grey[:, :, np.newaxis], 3, axis=2)

    return image


def randomHueSaturationValue(image, hue_shift_limit=(-180, 180),
                             sat_shift_limit=(-255, 255),
                             val_shift_limit=(-255, 255), u=0.5):
    if np.random.random() < u:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(image)
        hue_shift = np.random.uniform(hue_shift_limit[0], hue_shift_limit[1])
        h = cv2.add(h, hue_shift)
        sat_shift = np.random.uniform(sat_shift_limit[0], sat_shift_limit[1])
        s = cv2.add(s, sat_shift)
        val_shift = np.random.uniform(val_shift_limit[0], val_shift_limit[1])
        v = cv2.add(v, val_shift)
        image = cv2.merge((h, s, v))
        image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)

    return image

def randomShiftScaleRotate(image, mask,
                           shift_limit=(-0.0625, 0.0625),
                           scale_limit=(-0.1, 0.1),
                           rotate_limit=(-45, 45), aspect_limit=(0, 0),
                           borderMode=cv2.BORDER_CONSTANT, u=0.5):
    if np.random.random() < u:
        height, width, channel = image.shape

        angle = np.random.uniform(rotate_limit[0], rotate_limit[1])  # degree
        scale = np.random.uniform(1 + scale_limit[0], 1 + scale_limit[1])
        aspect = np.random.uniform(1 + aspect_limit[0], 1 + aspect_limit[1])
        sx = scale * aspect / (aspect ** 0.5)
        sy = scale / (aspect ** 0.5)
        dx = round(np.random.uniform(shift_limit[0], shift_limit[1]) * width)
        dy = round(np.random.uniform(shift_limit[0], shift_limit[1]) * height)

        cc = np.math.cos(angle / 180 * np.math.pi) * sx
        ss = np.math.sin(angle / 180 * np.math.pi) * sy
        rotate_matrix = np.array([[cc, -ss], [ss, cc]])

        box0 = np.array([[0, 0], [width, 0], [width, height], [0, height], ])
        box1 = box0 - np.array([width / 2, height / 2])
        box1 = np.dot(box1, rotate_matrix.T) + np.array([width / 2 + dx, height / 2 + dy])

        box0 = box0.astype(np.float32)
        box1 = box1.astype(np.float32)
        mat = cv2.getPerspectiveTransform(box0, box1)
        image = cv2.warpPerspective(image, mat, (width, height), flags=cv2.INTER_LINEAR, borderMode=borderMode,
                                    borderValue=(
                                        0, 0,
                                        0,))
        mask = cv2.warpPerspective(mask, mat, (width, height), flags=cv2.INTER_LINEAR, borderMode=borderMode,
                                   borderValue=(
                                       0, 0,
                                       0,))

    return image, mask


def randomHorizontalFlip(image, mask, u=0.5):
    if np.random.random() < u:
        image = cv2.flip(image, 1)
        mask = cv2.flip(mask, 1)

    return image, mask


# @threadsafe_generator
# def train_generator(input_path, idx, batch_size, img_size, aug=False):
#     while True:
#         for start in range(0, len(idx), batch_size):
#             x_batch = []
#             y_batch = []
#             end = min(start + batch_size, len(idx))
#             ids_train_batch = idx[start:end]
#             for id in ids_train_batch.values:
#                 img = cv2.imread(input_path + 'train/{}.jpg'.format(id))
#                 img = cv2.resize(img, (img_size, img_size))
#                 mask = cv2.imread(input_path + 'train_masks/{}_mask.png'.format(id), cv2.IMREAD_GRAYSCALE)
#                 mask = cv2.resize(mask, (img_size, img_size))
#                 if aug:
#                     img, mask = randomShiftScaleRotate(img, mask,
#                                                        shift_limit=(-0.0625, 0.0625),
#                                                        scale_limit=(-0.1, 0.1),
#                                                        rotate_limit=(-0, 0))
#                     img, mask = randomHorizontalFlip(img, mask)
#                 mask = np.expand_dims(mask, axis=2)
#                 x_batch.append(img)
#                 y_batch.append(mask)
#             x_batch = np.array(x_batch, np.float32) / 255
#             y_batch = np.array(y_batch, np.float32) / 255
#             yield x_batch, y_batch
#
#
# def valid_generator(input_path, ids_valid_split, batch_size, img_size):
#     while True:
#         for start in range(0, len(ids_valid_split), batch_size):
#             x_batch = []
#             y_batch = []
#             end = min(start + batch_size, len(ids_valid_split))
#             ids_valid_batch = ids_valid_split[start:end]
#             for id in ids_valid_batch.values:
#                 img = cv2.imread(input_path+'train/{}.jpg'.format(id))
#                 img = cv2.resize(img, (img_size, img_size))
#                 mask = cv2.imread(input_path+'train_masks/{}_mask.png'.format(id), cv2.IMREAD_GRAYSCALE)
#                 mask = cv2.resize(mask, (img_size, img_size))
#                 mask = np.expand_dims(mask, axis=2)
#                 x_batch.append(img)
#                 y_batch.append(mask)
#             x_batch = np.array(x_batch, np.float32) / 255
#             y_batch = np.array(y_batch, np.float32) / 255
#             yield x_batch, y_batch

def start_training(args):
    model_name = str(args.model)
    output_path = str(args.out_path) + '/'
    input_path = str(args.in_path) + '/'

    epochs = int(args.epochs)
    batch_size = int(args.batch_size)

    nb_classes = 1

    df_train = pd.read_csv(input_path + 'train_masks.csv')
    ids_train = df_train['img'].map(lambda s: s.split('.')[0])
    # print(ids_train)
    ids_train_split, ids_valid_split = train_test_split(ids_train, test_size=0.2, random_state=42)

    print('Training on {} samples'.format(len(ids_train_split)))
    print('Validating on {} samples'.format(len(ids_valid_split)))

    model = None
    image_size = 256
    # batch_size = 24
    model_path = 'weights/best_weights.hdf5'
    csv_path = output_path + 'train_log.csv'
    if model_name == 'unet128':
        model = get_unet_128()
        image_size = 128
        batch_size = 48
        model_path = output_path + 'Unet128_{val_loss:.4f}-{val_binary_accuracy:.4f}-{val_dice_loss:.5f}.hdf5'
        csv_path = output_path + 'Unet128.csv'
    # tested
    elif model_name == 'unet256':
        model_path = output_path + 'Unet256_{val_loss:.4f}-{val_binary_accuracy:.4f}-{val_dice_loss:.5f}.hdf5'
        csv_path = output_path + 'Unet256.csv'
        img_tuple = (256, 256, 3)
        batch_size = 24
        model = get_unet_256(input_shape=img_tuple, num_classes=nb_classes)
        model.compile(optimizer=optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True),
                      loss=weighted_loss,
                      metrics=['binary_accuracy', dice_loss])

        callbacks = [
                    #  EarlyStopping(monitor='val_dice_loss',
                    #                patience=8,
                    #                verbose=1,
                    #                min_delta=1e-05,
                    #                mode='max'),
                    #  ReduceLROnPlateau(monitor='val_dice_loss',
                    #                    factor=0.333,
                    #                    patience=3,
                    #                    verbose=1,
                    #                    epsilon=1e-05,
                    #                    mode='max',
                    #                    min_lr=1e-04),
                     ModelCheckpoint(monitor='val_dice_loss',
                                     filepath=model_path,
                                     save_best_only=True,
                                     save_weights_only=True,
                                     mode='max'),
                     TensorBoard(log_dir=output_path),
                     CSVLogger(filename=csv_path, separator=',', append=True),
                     LearningRateScheduler(unet_lr_scheduler)
                     ]
    # elif model_name == 'unet512':
    #     model = get_unet_512()
    #     image_size = 512
    #     batch_size = 12
    #     model_path = output_path + 'Unet512_{val_loss:.4f}-{val_binary_accuracy:.4f}-{val_dice_loss:.5f}.hdf5'
    #     csv_path = output_path + 'Unet512.csv'
    elif model_name == 'unet1024':
        model_path = output_path + 'Unet1024_best_model.hdf5'
        csv_path = output_path + 'Unet1024.csv'
        img_tuple = (256, 256, 3)
        # batch_size = 3
        model = get_unet_1024(input_shape=img_tuple, num_classes=nb_classes)
        model.compile(optimizer=utils.SGD(lr=1e-02,
                                          decay=1e-6,
                                          momentum=0.9,
                                          nesterov=True,
                                          accumulator=np.ceil(float(64) / float(batch_size))),
                      loss=weighted_loss,
                      metrics=['binary_accuracy', dice_loss])

        callbacks = [
                     EarlyStopping(monitor='val_dice_loss',
                                   patience=8,
                                   verbose=1,
                                   min_delta=1e-05,
                                   mode='max'),
                     ReduceLROnPlateau(monitor='val_dice_loss',
                                       factor=0.333,
                                       patience=3,
                                       verbose=1,
                                       epsilon=1e-05,
                                       mode='max',
                                       min_lr=1e-04),
                     ModelCheckpoint(monitor='val_dice_loss',
                                     filepath=model_path,
                                     save_best_only=True,
                                     save_weights_only=True,
                                     mode='max'),
                     TensorBoard(log_dir=output_path),
                     CSVLogger(filename=csv_path, separator=',', append=True),
                    #  LearningRateScheduler(unet_lr_scheduler)
                     ]

    elif model_name == 'unet':
        model_path = output_path + 'Unet_best_model.hdf5'
        csv_path = output_path + 'Unet.csv'
        img_tuple = (256, 256, 3)
        batch_size = 24
        model = get_unet(input_shape=img_tuple, num_classes=nb_classes)
        model.compile(optimizer=utils.SGD(lr=1e-02, decay=1e-6, momentum=0.9, nesterov=True, accumulator=1),
                      loss=weighted_loss,
                      metrics=['binary_accuracy', dice_loss])

        callbacks = [
                     EarlyStopping(monitor='val_dice_loss',
                                   patience=6,
                                   verbose=1,
                                   min_delta=1e-05,
                                   mode='max'),
                     ReduceLROnPlateau(monitor='val_dice_loss',
                                       factor=0.333,
                                       patience=3,
                                       verbose=1,
                                       epsilon=1e-05,
                                       mode='max',
                                       min_lr=3e-04),
                     ModelCheckpoint(monitor='val_dice_loss',
                                     filepath=model_path,
                                     save_best_only=True,
                                     save_weights_only=True,
                                     mode='max'),
                     TensorBoard(log_dir=output_path),
                     CSVLogger(filename=csv_path, separator=',', append=True),
                    #  LearningRateScheduler(unet_lr_scheduler)
                     ]
    elif model_name == 'modify_unet':
        model_path = output_path + 'Modify_Unet_best_model.hdf5'
        csv_path = output_path + 'Modify_Unet.csv'
        img_tuple = (512, 512, 3)
        # batch_size = 3
        model = get_modify_unet(input_shape=img_tuple, num_classes=nb_classes)

        if args.noweight:
            model.compile(optimizer=utils.SGD(lr=1e-02,
                                              decay=1e-6,
                                              momentum=0.9,
                                              nesterov=True,
                                              accumulator=np.ceil(float(15) / float(batch_size))),
                          loss=weighted_loss,
                          metrics=['binary_accuracy', dice_loss])
        else:
            model.compile(optimizer=utils.Adam(lr=1e-02,
                                               accumulator=np.ceil(float(15) / float(batch_size))),
                          loss=weighted_loss,
                          metrics=['binary_accuracy', dice_loss])

        callbacks = [
                     EarlyStopping(monitor='val_dice_loss',
                                   patience=8,
                                   verbose=1,
                                   min_delta=1e-05,
                                   mode='max'),
                     ReduceLROnPlateau(monitor='val_dice_loss',
                                       factor=0.333,
                                       patience=3,
                                       verbose=1,
                                       epsilon=1e-04,
                                       mode='max',
                                       min_lr=3e-04),
                     ModelCheckpoint(monitor='val_dice_loss',
                                     filepath=model_path,
                                     save_best_only=True,
                                     save_weights_only=True,
                                     mode='max'),
                     TensorBoard(log_dir=output_path),
                     CSVLogger(filename=csv_path, separator=',', append=True),
                    #  LearningRateScheduler(enet_lr_scheduler)
                     ]
    elif model_name == 'modify_unet_V2':
        model_path = output_path + 'Modify_Unet_best_model.hdf5'
        csv_path = output_path + 'Modify_Unet.csv'
        img_tuple = (1024, 1024, 3)
        # batch_size = 5
        model = get_modify_unet_V2(input_shape=img_tuple, num_classes=nb_classes)

        if args.noweight:
            model.compile(optimizer=utils.SGD(lr=1e-02,
                                              decay=1e-6,
                                              momentum=0.9,
                                              nesterov=True,
                                              accumulator=np.ceil(float(40) / float(batch_size))),
                          loss=weighted_loss,
                          metrics=['binary_accuracy', dice_loss])
        else:
            model.compile(optimizer=utils.SGD(lr=1e-02,
                                              decay=1e-6,
                                              momentum=0.9,
                                              nesterov=True,
                                              accumulator=np.ceil(float(40) / float(batch_size))),
                          loss=weighted_loss,
                          metrics=['binary_accuracy', dice_loss])

        callbacks = [
                     EarlyStopping(monitor='val_dice_loss',
                                   patience=8,
                                   verbose=1,
                                   min_delta=1e-05,
                                   mode='max'),
                     ReduceLROnPlateau(monitor='val_dice_loss',
                                       factor=0.333,
                                       patience=3,
                                       verbose=1,
                                       epsilon=1e-04,
                                       mode='max',
                                       min_lr=1e-04),
                     ModelCheckpoint(monitor='val_dice_loss',
                                     filepath=model_path,
                                     save_best_only=True,
                                     save_weights_only=True,
                                     mode='max'),
                     TensorBoard(log_dir=output_path),
                     CSVLogger(filename=csv_path, separator=',', append=True),
                    #  LearningRateScheduler(unet_lr_scheduler)
                     ]
    # elif model_name == 'segnet':
    #     model_path = output_path + 'SegNet_{val_loss:.4f}-{val_binary_accuracy:.4f}-{val_dice_loss:.5f}.hdf5'
    #     csv_path = output_path + 'SegNet.csv'
    #     image_size = 1024
    #     batch_size = 1
    #     model = SegNet(input_shape=(image_size, image_size, 3), classes=nb_classes)
    #     model.compile(optimizer=optimizers.Adam(lr=0.01), loss=bce_dice_loss, metrics=['binary_accuracy',
    #                                                                                     dice_loss])
    #
    # elif model_name == 'segnet_full':
    #     model_path = output_path + 'SegNet_FULL_{val_loss:.4f}-{val_binary_accuracy:.4f}-{val_dice_loss:.5f}.hdf5'
    #     csv_path = output_path + 'SegNet_FULL.csv'
    #     image_size = 1024
    #     batch_size = 2
    #     model = SegNet(input_shape=(image_size, image_size, 3), classes=nb_classes)
    #     model.compile(optimizer=optimizers.Adam(lr=0.01), loss=bce_dice_loss, metrics=['binary_accuracy',
    #                                                                                     dice_loss])
    # elif model_name == 'enet_unpooling':
    #     model_path = output_path + 'Enet_unpooling_{val_loss:.4f}-{val_binary_accuracy:.4f}.hdf5'
    #     csv_path = output_path + 'Enet_unpooling.csv'
    #     image_size = int(256)
    #     model = get_enet_unpooling(input_shape=(image_size, image_size, 3), classes=nb_classes)
    #     model.compile(optimizer=optimizers.Adam(lr=0.01), loss=losses.binary_crossentropy, metrics=['binary_accuracy',
    #                                                                                       dice_loss])
    elif model_name == 'enet_naive_upsampling':
        model_path = output_path + 'Enet_naive_upsampling_best_model.hdf5'
        csv_path = output_path + 'Enet_naive_upsampling.csv'
        img_tuple = (512, 512, 3)
        # batch_size = 6
        model = get_enet_naive_upsampling(input_shape=img_tuple, classes=nb_classes)


        if args.noweight:
            model.compile(optimizer=utils.SGD(lr=1e-02,
                                              decay=1e-6,
                                              momentum=0.9,
                                              nesterov=True,
                                              accumulator=np.ceil(float(14) / float(batch_size))),
                          loss=weighted_loss,
                          metrics=['binary_accuracy', dice_loss])
        else:
            model.compile(optimizer=optimizers.Adam(lr=1e-02),
                          loss=weighted_loss,
                          metrics=['binary_accuracy', dice_loss])

        callbacks = [
                     EarlyStopping(monitor='val_dice_loss',
                                   patience=8,
                                   verbose=1,
                                   min_delta=1e-05,
                                   mode='max'),
                     ReduceLROnPlateau(monitor='val_dice_loss',
                                       factor=0.333,
                                       patience=3,
                                       verbose=1,
                                       epsilon=1e-05,
                                       mode='max',
                                       min_lr=3e-04),
                     ModelCheckpoint(monitor='val_dice_loss',
                                     filepath=model_path,
                                     save_best_only=True,
                                     save_weights_only=True,
                                     mode='max'),
                     TensorBoard(log_dir=output_path),
                     CSVLogger(filename=csv_path, separator=',', append=True),
                    #  LearningRateScheduler(enet_lr_scheduler)
                     ]

    # elif model_name == 'densenet_fc':
    #     model_path = output_path + 'DenseNet_FC_{val_loss:.4f}-{val_binary_accuracy:.4f}-{val_dice_loss:.5f}.hdf5'
    #     csv_path = output_path + 'DenseNet_FC.csv'
    #     image_size = 256
    #     batch_size = 8
    #     base_model = DenseNetFCN(input_shape=(image_size, image_size, 3), classes=nb_classes, activation='sigmoid', include_top=False, batchsize=batch_size)
    #     classify = Conv2D(nb_classes, (1, 1), activation='sigmoid')(base_model.output)
    #     model = Model(inputs=base_model.input, outputs=classify)
    #     model.compile(optimizer=optimizers.Adam(lr=0.01), loss=bce_dice_loss, metrics=['binary_accuracy',
    #                                                                                     dice_loss])
    elif model_name == 'linknet':
        model_path = output_path + 'LinkNet_best_model.hdf5'
        csv_path = output_path + 'LinkNet.csv'
        img_tuple = (1024, 1024, 3)
        batch_size = 6
        base_model = LinkNet(input_shape=img_tuple, classes=nb_classes)
        classify = Conv2D(nb_classes, (1, 1), activation='sigmoid')(base_model.output)
        model = Model(inputs=base_model.input, outputs=classify)
        model.compile(optimizer=utils.SGD(lr=1e-02, decay=1e-6, momentum=0.9, nesterov=True, accumulator=4),
                      loss=weighted_loss,
                      metrics=['binary_accuracy', dice_loss])


        callbacks = [
                     EarlyStopping(monitor='val_dice_loss',
                                   patience=8,
                                   verbose=1,
                                   min_delta=1e-05,
                                   mode='max'),
                     ReduceLROnPlateau(monitor='val_dice_loss',
                                       factor=0.333,
                                       patience=3,
                                       verbose=1,
                                       epsilon=1e-05,
                                       mode='max',
                                       min_lr=1e-04),
                     ModelCheckpoint(monitor='val_dice_loss',
                                     filepath=model_path,
                                     save_best_only=True,
                                     save_weights_only=True,
                                     mode='max'),
                     TensorBoard(log_dir=output_path),
                     CSVLogger(filename=csv_path, separator=',', append=True),
                    #  LearningRateScheduler(linknet_lr_scheduler)
                     ]

    elif model_name == 'linknet_res34':
        model_path = output_path + 'LinkNet_Res34_best_model.hdf5'
        csv_path = output_path + 'LinkNet_Res34.csv'
        img_tuple = (1024, 1024, 3)
        # batch_size = 12
        base_model = LinkNet_Res34(input_shape=img_tuple, classes=nb_classes)
        classify = Conv2D(nb_classes, (1, 1), activation='sigmoid')(base_model.output)
        model = Model(inputs=base_model.input, outputs=classify)
        model.compile(optimizer=utils.SGD(lr=1e-02,
                                          decay=1e-6,
                                          momentum=0.9,
                                          nesterov=True,
                                          accumulator=np.ceil(float(84) / float(batch_size))),
                      loss=weighted_loss_256,
                      metrics=['binary_accuracy', dice_loss])


        callbacks = [
                     EarlyStopping(monitor='val_dice_loss',
                                   patience=8,
                                   verbose=1,
                                   min_delta=1e-05,
                                   mode='max'),
                     ReduceLROnPlateau(monitor='val_dice_loss',
                                       factor=0.333,
                                       patience=3,
                                       verbose=1,
                                       epsilon=1e-05,
                                       mode='max',
                                       min_lr=1e-04),
                     ModelCheckpoint(monitor='val_dice_loss',
                                     filepath=model_path,
                                     save_best_only=True,
                                     save_weights_only=True,
                                     mode='max'),
                     TensorBoard(log_dir=output_path),
                     CSVLogger(filename=csv_path, separator=',', append=True),
                    #  LearningRateScheduler(unet_lr_scheduler)
                     ]

    elif model_name == 'linknet_res50':
        model_path = output_path + 'LinkNet_Res50_{val_loss:.4f}-{val_binary_accuracy:.4f}-{val_dice_loss:.5f}.hdf5'
        csv_path = output_path + 'LinkNet_Res50.csv'
        img_tuple = (1280, 1280, 3)
        batch_size = 2
        base_model = LinkNet_Res50(input_shape=img_tuple, classes=nb_classes)
        classify = Conv2D(nb_classes, (1, 1), activation='sigmoid')(base_model.output)
        model = Model(inputs=base_model.input, outputs=classify)
        model.compile(optimizer=utils.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True, accumulator=10),
                      loss=weighted_loss,
                      metrics=['binary_accuracy', dice_loss])


        callbacks = [
                    #  EarlyStopping(monitor='val_dice_loss',
                    #                patience=8,
                    #                verbose=1,
                    #                min_delta=1e-05,
                    #                mode='max'),
                    #  ReduceLROnPlateau(monitor='val_dice_loss',
                    #                    factor=0.333,
                    #                    patience=3,
                    #                    verbose=1,
                    #                    epsilon=1e-05,
                    #                    mode='max',
                    #                    min_lr=1e-04),
                     ModelCheckpoint(monitor='val_dice_loss',
                                     filepath=model_path,
                                     save_best_only=True,
                                     save_weights_only=True,
                                     mode='max'),
                     TensorBoard(log_dir=output_path),
                     CSVLogger(filename=csv_path, separator=',', append=True),
                     LearningRateScheduler(unet_lr_scheduler)
                     ]

    if model:
        train_batch_gen = iter(Batch_Generator(input_path=input_path,
                                               idx=ids_train_split,
                                               batch_size=batch_size,
                                               img_tuple=img_tuple,
                                               aug=True,
                                               sat=args.sat,
                                               hue=args.hue,
                                               val=args.val,
                                               rotate=args.rotate,
                                               gray=args.gray,
                                               contrast=args.contrast,
                                               brightness=args.brightness,
                                               shear=args.shear,
                                               shift=args.shift,
                                               scale=args.scale,
                                               lowq=args.lowq))

        valid_batch_gen = iter(Batch_Generator(input_path=input_path,
                                               idx=ids_valid_split,
                                               batch_size=batch_size,
                                               img_tuple=img_tuple,
                                               aug=False))

        print('Start to train model hue %r, sat %r, val %r, rotate %r, gray %r, contrast %r, brightness %r, shear %r, shift %r, scale %r, lowq %r, none weight %r'
                                                                                                        % (args.hue,
                                                                                                           args.sat,
                                                                                                           args.val,
                                                                                                           args.rotate,
                                                                                                           args.gray,
                                                                                                           args.contrast,
                                                                                                           args.brightness,
                                                                                                           args.shear,
                                                                                                           args.shift,
                                                                                                           args.scale,
                                                                                                           args.lowq,
                                                                                                           args.noweight,))

        model.fit_generator(generator=train_batch_gen,
                            steps_per_epoch=np.ceil(float(len(ids_train_split)) / float(batch_size)),
                            epochs=epochs,
                            callbacks=callbacks,
                            validation_data=valid_batch_gen,
                            validation_steps=np.ceil(float(len(ids_valid_split)) / float(batch_size)),
                            workers=6)


        # model.fit_generator(generator=train_generator(input_path=input_path,
        #                                               idx=ids_train_split,
        #                                               batch_size=batch_size,
        #                                               img_size=image_size,
        #                                               aug=True),
        #                     steps_per_epoch=np.ceil(float(len(ids_train_split)) / float(batch_size)),
        #                     epochs=epochs,
        #                     callbacks=callbacks,
        #                     validation_data=train_generator(input_path=input_path,
        #                                                     idx=ids_valid_split,
        #                                                     batch_size=batch_size,
        #                                                     img_size=image_size,
        #                                                     aug=False),
        #                     validation_steps=np.ceil(float(len(ids_valid_split)) / float(batch_size)),
        #                     workers=1)

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help='Model name')

    parser.add_argument('--out_path', help='Output path', default='output/')
    parser.add_argument('--in_path', help='input dir', default='input/')

    parser.add_argument('--epochs', help='the number of epoch', default=50)
    parser.add_argument('--batch_size', help='Batch size', default=24)

    parser.add_argument("--hue", type=str2bool, nargs='?',
                        const=True, default=False,
                        help="Activate nice mode.")

    parser.add_argument("--sat", type=str2bool, nargs='?',
                        const=True, default=False,
                        help="Activate nice mode.")

    parser.add_argument("--val", type=str2bool, nargs='?',
                        const=True, default=False,
                        help="Activate nice mode.")

    parser.add_argument("--rotate", type=str2bool, nargs='?',
                        const=True, default=False,
                        help="Activate nice mode.")

    parser.add_argument("--gray", type=str2bool, nargs='?',
                        const=True, default=False,
                        help="Activate nice mode.")

    parser.add_argument("--contrast", type=str2bool, nargs='?',
                        const=True, default=False,
                        help="Activate nice mode.")

    parser.add_argument("--brightness", type=str2bool, nargs='?',
                        const=True, default=False,
                        help="Activate nice mode.")

    parser.add_argument("--shear", type=str2bool, nargs='?',
                        const=True, default=False,
                        help="Activate nice mode.")

    parser.add_argument("--shift", type=str2bool, nargs='?',
                        const=True, default=False,
                        help="Activate nice mode.")

    parser.add_argument("--scale", type=str2bool, nargs='?',
                        const=True, default=False,
                        help="Activate nice mode.")

    parser.add_argument("--lowq", type=str2bool, nargs='?',
                        const=True, default=False,
                        help="Activate nice mode.")

    parser.add_argument("--noweight", type=str2bool, nargs='?',
                        const=True, default=False,
                        help="Activate nice mode.")

    args = parser.parse_args()

    start_training(args)

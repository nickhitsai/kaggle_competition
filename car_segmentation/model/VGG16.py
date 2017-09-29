from keras.models import Model
from keras.layers import Conv2D, MaxPooling2D, concatenate
from keras.layers.core import Dense, Reshape, Flatten, Dropout
from keras import backend as K
from keras import applications, losses
from keras import optimizers


def dice_loss(y_true, y_pred):
    smooth = 1.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def bce_dice_loss(y_true, y_pred):
    return losses.binary_crossentropy(y_true, y_pred) + (1 - dice_loss(y_true, y_pred))

def get_vgg16(image_size=256, num_classes=1):
    base_model = applications.vgg16.VGG16(weights='imagenet',
                                          include_top=False,
                                          input_shape=(image_size, image_size, 3))

    # for layer in base_model.layers:
    #     layer.trainable = False

    # add a global spatial average pooling layer
    x = Flatten()(base_model.output)
    x = Dense(4096, activation='relu')(x)
    x = Dense(4096, activation='relu')(x)
    # predictions = Conv2D(num_classes, (1, 1), activation='sigmoid')(base_model.output)
    predictions = Dense(num_classes, activation='sigmoid')(x)

    # this is the model we will train
    model = Model(inputs=base_model.input, outputs=predictions)

    opt = optimizers.Adam(lr=3e-05)

    model.compile(optimizer=opt, loss=dice_loss, metrics=['binary_accuracy',
                                                          losses.binary_crossentropy])

    return model

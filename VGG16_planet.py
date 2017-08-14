from __future__ import print_function

import datetime, sys, random, os, csv
import keras, glob, cv2
import numpy as np

from keras.models import Sequential, Model, load_model
from keras.layers import Conv2D, MaxPooling2D, concatenate
from keras.layers.pooling import GlobalAveragePooling2D
from keras.layers.core import Dense, Reshape, Flatten, Dropout
from keras import backend as K
from keras import applications, losses
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, CSVLogger, LearningRateScheduler
from keras.preprocessing.image import ImageDataGenerator

import tensorflow as tf

from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import fbeta_score

epochs = 20
nfolds = 5
batch_size = 48
samples_valid_per_epoch = 172 # 32400 / (epochs*batch_size)
samples_train_per_epoch = samples_valid_per_epoch*4

num_classes = 17
filepath = '../train-jpg/'

root_path = '../results/VGG16/f_beta_20_epoch/'

callbackpath =  root_path + 'VGG16_{val_loss:.4f}-{val_binary_accuracy:.4f}-{val_recall:.4f}.hdf5'
high_recall_callbackpath =  root_path + 'VGG16_recall_{val_loss:.4f}-{val_binary_accuracy:.4f}-{val_recall:.4f}.hdf5'

csv_filepath = '../train_v2.csv'
logger_filepath = root_path + 'VGG16_log.csv'
image_size = 256

label = {'agriculture':0,
         'artisinal_mine':1,
         'bare_ground':2,
         'blooming':3,
         'blow_down':4,
         'clear':5,
         'cloudy':6,
         'conventional_mine':7,
         'cultivation':8,
         'habitation':9,
         'haze':10,
         'partly_cloudy':11,
         'primary':12,
         'road':13,
         'selective_logging':14,
         'slash_burn':15,
         'water':16}




#
# Following two functions come from https://github.com/fchollet/keras/commit/a56b1a55182acf061b1eb2e2c86b48193a0e88f7
#

def precision(y_true, y_pred):
    """Precision metric.

    Only computes a batch-wise average of precision.

    Computes the precision, a metric for multi-label classification of
    how many selected items are relevant.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def recall(y_true, y_pred):
    """Recall metric.

    Only computes a batch-wise average of recall.

    Computes the recall, a metric for multi-label classification of
    how many relevant items are selected.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


# def precision_4_loss(y_true, y_pred):
#     """Precision metric.
#
#     Only computes a batch-wise average of precision.
#
#     Computes the precision, a metric for multi-label classification of
#     how many selected items are relevant.
#     """
#     true_positives = K.sum(tf.floor(K.clip(y_true * y_pred, 0, 1) + 0.5))
#     predicted_positives = K.sum(tf.floor(K.clip(y_pred, 0, 1) + 0.5))
#     precision = true_positives / (predicted_positives + K.epsilon())
#     return precision
#
#
# def recall_4_loss(y_true, y_pred):
#     """Recall metric.
#
#     Only computes a batch-wise average of recall.
#
#     Computes the recall, a metric for multi-label classification of
#     how many relevant items are selected.
#     """
#     true_positives = K.sum(tf.floor(K.clip(y_true * y_pred, 0, 1) + 0.5))
#     possible_positives = K.sum(tf.floor(K.clip(y_true, 0, 1) + 0.5))
#     recall = true_positives / (possible_positives + K.epsilon())
#     return recall
#
# def fbeta_sklearn(y_true, y_pred):
#     return fbeta_score(y_true, np.round(y_pred), beta=2)

def fbeta_keras(y_true, y_pred, threshold_shift=0):
    beta = 2

    # just in case of hipster activation at the final layer
    y_pred = K.clip(y_pred, 0, 1)

    # shifting the prediction threshold from .5 if needed
    y_pred_bin = K.round(y_pred + threshold_shift)

    tp = K.sum(K.round(y_true * y_pred_bin)) + K.epsilon()
    fp = K.sum(K.round(K.clip(y_pred_bin - y_true, 0, 1)))
    fn = K.sum(K.round(K.clip(y_true - y_pred, 0, 1)))

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)

    beta_squared = beta ** 2
    return (beta_squared + 1) * (precision * recall) / (beta_squared * precision + recall + K.epsilon())
#
# def fbeta_tensor(y_true, y_pred, threshold_shift=0.5):
#     beta = 2
#
#     # just in case of hipster activation at the final layer
#     y_pred = K.clip(y_pred, 0, 1)
#
#     # shifting the prediction threshold from .5 if needed
#     y_pred_bin = tf.floor(y_pred + threshold_shift)
#
#     tp = K.sum(tf.floor(y_true * y_pred_bin  + threshold_shift)) + K.epsilon()
#     fp = K.sum(tf.floor(K.clip(y_pred_bin - y_true, 0, 1)  + threshold_shift))
#     fn = K.sum(tf.floor(K.clip(y_true - y_pred, 0, 1)  + threshold_shift))
#
#     precision = tp / (tp + fp)
#     recall = tp / (tp + fn)
#
#     beta_squared = beta ** 2
#     x = (beta_squared + 1) * (precision * recall) / (beta_squared * precision + recall + K.epsilon())
#     return (-1) * K.log(x)

def fbeta_tensor_loss(y_true, y_pred, threshold_shift=0.5):

    beta = 2

    # just in case of hipster activation at the final layer
    y_pred = K.clip(y_pred, 0, 1)

    # shifting the prediction threshold from .5 if needed
    y_pred_bin = tf.floor(y_pred + threshold_shift)

    tp = K.sum(tf.floor(y_true * y_pred_bin  + threshold_shift)) + K.epsilon()
    fp = K.sum(tf.floor(K.clip(y_pred_bin - y_true, 0, 1)  + threshold_shift))
    fn = K.sum(tf.floor(K.clip(y_true - y_pred, 0, 1)  + threshold_shift))

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)

    beta_squared = beta ** 2
    x = (beta_squared + 1) * (precision * recall) / (beta_squared * precision + recall + K.epsilon())
    return (0.4) * (-1) * K.log(x) + (1) * losses.binary_crossentropy(y_true, y_pred)

def custom_precision_recall_loss(y_true, y_pred, weights=0.2):
    return (-1) * K.log(recall_4_loss(y_true, y_pred)) + weights * (-1) * K.log(precision_4_loss(y_true, y_pred)) + (1e-5) * losses.binary_crossentropy(y_true, y_pred)

def custom_binary_crossentropy_precision_recall_loss(y_true, y_pred, weights=0.2):
    return weights * (-1) * K.log(recall_4_loss(y_true, y_pred)) + losses.binary_crossentropy(y_true, y_pred)

def custom_binary_crossentropy_loss(y_true, y_pred):
    return losses.binary_crossentropy(y_true, y_pred)

def learning_rate_scheduler(epoch_index):
    lr = 1e-07

    if epoch_index < 6:                         # 1-8
        lr = 3e-05
        # lr = 1e-03
    elif epoch_index < 10 and epoch_index > 5:  # 9-14
        lr = 1e-05
    elif epoch_index < 13 and epoch_index > 9:  # 15-18
        lr = 3e-06
    elif epoch_index < 15 and epoch_index > 12: # 19-20
        lr = 1e-06
    elif epoch_index < 16 and epoch_index > 14: # 15
        lr = 3e-07
    else:                                       # 15
        lr = 1e-07

    return lr

def batch_generator(files, categories, labelencoder, batch_size, augment):
    random.shuffle(files)
    number_of_batches = np.floor(len(files) / batch_size)
    counter = 0

    while True:
        batch_files = files[batch_size * counter:batch_size * (counter + 1)]
        image_list = []
        label_list = []
        for f in batch_files:
            image = cv2.imread(f)
            if image_size != 256:
                image = cv2.resize(image, (image_size, image_size))

            if augment:
                # flip along x axis
                if random.randint(0, 1) == 1:
                    image = cv2.flip(image, 0)

                # flip along y axis
                if random.randint(0, 1) == 1:
                    image = cv2.flip(image, 1)

                # grey scale
                # if random.randint(0, 1) == 1:
                #     grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                #     image = np.repeat(grey[:, :, np.newaxis], 3, axis=2)

                _randint = random.randint(0, 3)

                if _randint == 1:
                    num_rows, num_cols = image.shape[:2]
                    rotation_matrix = cv2.getRotationMatrix2D((num_cols/2, num_rows/2), 90, 1)
                    image = cv2.warpAffine(image, rotation_matrix, (num_cols, num_rows))
                elif _randint == 2:
                    num_rows, num_cols = image.shape[:2]
                    rotation_matrix = cv2.getRotationMatrix2D((num_cols/2, num_rows/2), 180, 1)
                    image = cv2.warpAffine(image, rotation_matrix, (num_cols, num_rows))
                elif _randint == 3:
                    num_rows, num_cols = image.shape[:2]
                    rotation_matrix = cv2.getRotationMatrix2D((num_cols/2, num_rows/2), 270, 1)
                    image = cv2.warpAffine(image, rotation_matrix, (num_cols, num_rows))

            # image = image.transpose((2,0,1))
            image = np.array(image)
            image = image / 255.0

            # f.split('/')[-1].split('.')[0] would get something like 'train_1'
            # than pass it to dict to get the categories
            clss = labelencoder.transform(categories[f.split('/')[-1].split('.')[0]])
            label = [0] * num_classes

            # This is for changing elements in list using a list
            label = np.array(label)
            label[clss] = 1

            image_list.append(image.astype(np.float32))
            label_list.append(label)
        counter += 1
        image_list = np.array(image_list)
        label_list = np.array(label_list)

        yield (image_list, label_list)

        if counter >= number_of_batches:
            random.shuffle(files)
            counter = 0

def train_model(model, num_fold, train_idx, test_idx, files, categories, labelencoder):
    train_images_list = []
    valid_images_list = []
    for i in train_idx:
        train_images_list.append(files[i])
    for i in test_idx:
        valid_images_list.append(files[i])

    start = datetime.datetime.now()

    callbacks = [
        ModelCheckpoint(callbackpath, monitor='val_loss', verbose=0, save_best_only=True, mode='min'),
        ModelCheckpoint(high_recall_callbackpath, monitor='val_recall', verbose=0, save_best_only=True, mode='max'),
        # ReduceLROnPlateau(monitor='val_loss', factor=0.33, patience=2, verbose=0, mode='min', epsilon=0.0001, cooldown=0, min_lr=1e-08),
        CSVLogger(filename=logger_filepath, separator=',', append=True),
        LearningRateScheduler(learning_rate_scheduler)
    ]

    history = model.fit_generator(generator=batch_generator(train_images_list, categories, labelencoder, batch_size, True),
                                  steps_per_epoch=samples_train_per_epoch,
                                  epochs=epochs,
                                  validation_data=batch_generator(valid_images_list, categories, labelencoder, batch_size, True),
                                  validation_steps=samples_valid_per_epoch,
                                  callbacks=callbacks)

    min_loss = min(history.history['val_loss'])
    print('Minimum loss for given fold: ', min_loss)
    print('Cost time: %.2f minutes' % float((datetime.datetime.now() - start) / datetime.timedelta(minutes=1)))
    # model.save_weights(model_name)
    return min_loss

def model_generator():
    base_model = applications.vgg16.VGG16(weights='imagenet',
                                          include_top=False,
                                          input_shape=(image_size, image_size, 3))

    # for layer in base_model.layers:
    #     layer.trainable = False

    # add a global spatial average pooling layer
    x = Flatten()(base_model.output)
    x = Dense(1024, activation='relu')(x)
    x = Dense(1024, activation='relu')(x)
    # x = Dense(classes, activation='softmax', name='predictions')(x)
    predictions = Dense(num_classes, activation='sigmoid')(x)

    # this is the model we will train
    model = Model(inputs=base_model.input, outputs=predictions)

    # opt = keras.optimizers.RMSprop(lr=0.00003, rho=0.9, epsilon=1e-08, decay=1e-6)
    # opt = keras.optimizers.SGD(lr=0.00003, decay=1e-6, momentum=0.9, nesterov=True) # too slow
    opt = keras.optimizers.Adam(lr=3e-05)

    model.compile(optimizer=opt, loss=fbeta_tensor_loss, metrics=['binary_accuracy',
                                                                                recall,
                                                                                precision])

    print(model.metrics_names)

    return model

def model_fine_tuner(model_path):
    model = load_model(model_path, custom_objects={
                                                #    "fbeta_tensor_loss": fbeta_tensor_loss,
                                                #    "fbeta_tensor": fbeta_tensor,
                                                #    "fbeta_keras": fbeta_keras,
                                                #    "recall_4_loss": recall_4_loss,
                                                   "recall": recall,
                                                #    "custom_precision_recall_loss": custom_precision_recall_loss,
                                                #    "precision_4_loss": precision_4_loss,
                                                   "precision": precision,
                                                #    "custom_binary_crossentropy_precision_recall_loss": custom_binary_crossentropy_precision_recall_loss,
                                                   "custom_binary_crossentropy_loss": custom_binary_crossentropy_loss
                                                   })

    return model.get_weights()

def category_parser(csv_filepath):
    category_dict = {}
    le = LabelEncoder()
    with open(csv_filepath, 'r') as f:
        reader = csv.reader(f)
        next(reader, None)  # skip the headers
        all_text_list = []
        for row in reader:
            tmp_text_list = row[1].split(' ')
            category_dict[row[0]] = tmp_text_list

            for t in tmp_text_list:
                if t not in all_text_list:
                    all_text_list.append(t)

        if len(all_text_list) != num_classes:
            print("all_text_list not match num_classes")
            sys.exit(0)

        le.fit(all_text_list)

    print(le.classes_)

    return category_dict, le

def main():
    files = glob.glob(filepath + '*.jpg')
    categories, labelencoder = category_parser(csv_filepath)
    model_path = '/home/fiiser/Documents/ML/workspace/planet/1080ti_area/VGG16/warm_up_3_epoch_1e-03/VGG16_0.1404-0.9452-0.7801.hdf5'
    init_weights = model_fine_tuner(model_path)

    model = model_generator()
    # init_weights = model.get_weights()

    num_fold = 0
    sum_score = 0
    kf = KFold(n_splits=nfolds, shuffle=True)
    for train_idx, test_idx in kf.split(files):
        model.set_weights(init_weights)

        num_fold += 1
        print('Start KFold number {} from {}'.format(num_fold, nfolds))
        print('Split train: ', len(train_idx))
        print('Split valid: ', len(test_idx))

        score = train_model(model, num_fold, train_idx, test_idx, files, categories, labelencoder)
        sum_score += score

    # score = train_model(model)
    print('loss: {}'.format(sum_score))

if __name__ == '__main__':
    main()

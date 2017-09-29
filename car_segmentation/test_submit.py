import cv2, argparse
import numpy as np
import pandas as pd
from tqdm import tqdm

from keras import losses, optimizers
from keras.layers import Conv2D, pooling
from keras.models import Model

from model.u_net import get_unet_128, get_unet_256, get_unet_512, get_unet_1024, dice_loss, bce_dice_loss, get_init_unet, get_unet, get_modify_unet, get_modify_unet_V2
from model.segnet import SegNet
from model.enet import get_enet_unpooling, get_enet_naive_upsampling
from model.linknet import LinkNet, LinkNet_Res34, LinkNet_Res50

from train import weighted_loss, str2bool

import utils

# https://www.kaggle.com/stainsby/fast-tested-rle
def run_length_encode(mask):
    '''
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    '''
    inds = mask.flatten()
    runs = np.where(inds[1:] != inds[:-1])[0] + 2
    runs[1::2] = runs[1::2] - runs[:-1:2]
    rle = ' '.join([str(r) for r in runs])
    return rle


def predict(args):
    nb_classes = 1

    orig_width = 1918
    orig_height = 1280

    model_name = str(args.model)
    model_weight = str(args.model_weight)

    output_path = str(args.out_path)
    input_path = str(args.in_path)

    batch_size = int(args.batchsize)
    image_size = int(args.image_size)

    df_test = pd.read_csv(input_path+'/sample_submission.csv')
    ids_test = df_test['img'].map(lambda s: s.split('.')[0])

    image_size = 256
    threshold = 0.5

    if model_name == 'unet128':
        image_size = 128
        batch_size = 48
        model = get_unet_128()
        model.load_weights(filepath=model_weight)
    # tested
    # elif model_name == 'unet256':
    #     image_size = 256
    #     batch_size = 24
    #     model = get_unet_256()
    #     model.load_weights(filepath=model_weight)
    # elif model_name == 'unet512':
    #     image_size = 512
    #     batch_size = 12
    #     model = get_unet_512()
    #     model.load_weights(filepath=model_weight)
    elif model_name == 'unet1024':
        img_tuple = (256, 256, 3)
        batch_size = 64
        model = get_unet_1024(input_shape=img_tuple, num_classes=nb_classes)
        model.compile(optimizer=utils.SGD(lr=1e-02,
                                          decay=1e-6,
                                          momentum=0.9,
                                          nesterov=True,
                                          accumulator=np.ceil(float(20) / float(batch_size))),
                      loss=weighted_loss,
                      metrics=['binary_accuracy', dice_loss])
        model.load_weights(filepath=model_weight)
    # elif model_name == 'segnet':
    #     image_size = 256
    #     batch_size = 28
    #     model = SegNet(input_shape=(image_size, image_size, 3), classes=nb_classes)
    #     model.compile(optimizer=optimizers.Adam(lr=0.01), loss=bce_dice_loss, metrics=['binary_accuracy',
    #                                                                                     dice_loss])
    #
    #     model.load_weights(filepath=model_weight)
    # elif model_name == 'enet_unpooling':
    #     model_path = output_path + 'Enet_unpooling_{val_loss:.4f}-{val_binary_accuracy:.4f}.hdf5'
    #     csv_path = output_path + 'Enet_unpooling.csv'
    #     image_size = int(256)
    #     model = get_enet_unpooling(input_shape=(image_size, image_size, 3), classes=nb_classes)
    #     model.compile(optimizer=optimizers.Adam(lr=0.01), loss=losses.binary_crossentropy, metrics=['binary_accuracy',
    #                                                                                       dice_loss])
    elif model_name == 'enet_naive_upsampling':
        img_tuple = (512, 512, 3)
        batch_size = 4
        model = get_enet_naive_upsampling(input_shape=img_tuple, classes=nb_classes)
        model.compile(optimizer=utils.SGD(lr=1e-02, decay=1e-6, momentum=0.9, nesterov=True, accumulator=5),
                      loss=weighted_loss,
                      metrics=['binary_accuracy', dice_loss])
        model.load_weights(filepath=model_weight)

    # elif model_name == 'init_unet':
    #     img_tuple = (1024, 1024, 3)
    #     batch_size = 6
    #     model = get_init_unet(input_shape=img_tuple, num_classes=nb_classes)
    #     model.compile(optimizer=utils.SGD(lr=1e-02, decay=1e-6, momentum=0.9, nesterov=True, accumulator=5),
    #                   loss=weighted_loss,
    #                   metrics=['binary_accuracy', dice_loss])
    #     model.load_weights(filepath=model_weight)
    #
    # elif model_name == 'unet':
    #     img_tuple = (256, 256, 3)
    #     batch_size = 24
    #     model = get_unet(input_shape=img_tuple, num_classes=nb_classes)
    #     model.compile(optimizer=utils.SGD(lr=1e-02, decay=1e-6, momentum=0.9, nesterov=True, accumulator=5),
    #                   loss=weighted_loss,
    #                   metrics=['binary_accuracy', dice_loss])
    #     model.load_weights(filepath=model_weight)

    elif model_name == 'modify_unet':
        img_tuple = (512, 512, 3)
        batch_size = 5
        model = get_modify_unet(input_shape=img_tuple, num_classes=nb_classes)
        model.compile(optimizer=utils.SGD(lr=1e-02, decay=1e-6, momentum=0.9, nesterov=True, accumulator=5),
                      loss=weighted_loss,
                      metrics=['binary_accuracy', dice_loss])
        model.load_weights(filepath=model_weight)

    # elif model_name == 'modify_unet_V2':
    #     img_tuple = (1024, 1024, 3)
    #     batch_size = 4
    #     model = get_modify_unet_V2(input_shape=img_tuple, num_classes=nb_classes)
    #     model.compile(optimizer=utils.SGD(lr=1e-02, decay=1e-6, momentum=0.9, nesterov=True, accumulator=5),
    #                   loss=weighted_loss,
    #                   metrics=['binary_accuracy', dice_loss])
    #     model.load_weights(filepath=model_weight)
    #
    # elif model_name == 'linknet':
    #     img_tuple = (1024, 1024, 3)
    #     batch_size = 6
    #
    #     base_model = LinkNet(input_shape=img_tuple, classes=nb_classes)
    #     classify = Conv2D(nb_classes, (1, 1), activation='sigmoid')(base_model.output)
    #     model = Model(inputs=base_model.input, outputs=classify)
    #     model.compile(optimizer=utils.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True, accumulator=10),
    #                   loss=weighted_loss,
    #                   metrics=['binary_accuracy', dice_loss])
    #
    #     model.load_weights(filepath=model_weight)

    elif model_name == 'linknet_res34':
        img_tuple = (256, 256, 3)
        batch_size = 84

        base_model = LinkNet_Res34(input_shape=img_tuple, classes=nb_classes)
        classify = Conv2D(nb_classes, (1, 1), activation='sigmoid')(base_model.output)
        model = Model(inputs=base_model.input, outputs=classify)
        model.compile(optimizer=utils.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True, accumulator=10),
                      loss=weighted_loss,
                      metrics=['binary_accuracy', dice_loss])

        model.load_weights(filepath=model_weight)

    # elif model_name == 'linknet_res50':
    #     img_tuple = (1280, 1280, 3)
    #     batch_size = 4
    #
    #     base_model = LinkNet_Res50(input_shape=img_tuple, classes=nb_classes)
    #     classify = Conv2D(nb_classes, (1, 1), activation='sigmoid')(base_model.output)
    #     model = Model(inputs=base_model.input, outputs=classify)
    #     model.compile(optimizer=utils.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True, accumulator=10),
    #                   loss=weighted_loss,
    #                   metrics=['binary_accuracy', dice_loss])
    #
    #     model.load_weights(filepath=model_weight)

    names = []
    for id in ids_test:
        names.append('{}.jpg'.format(id))

    rles = []

    print('Predicting on {} samples with batch_size = {}...'.format(len(ids_test), batch_size))
    print('Conditions low quality %r' %(args.lowq,))
    for start in tqdm(range(0, len(ids_test), batch_size)):
        x_batch = []
        end = min(start + batch_size, len(ids_test))
        ids_test_batch = ids_test[start:end]
        for id in ids_test_batch.values:
            if args.lowq:
                img = cv2.imread(input_path+'test/{}.jpg'.format(id))
            else:
                img = cv2.imread(input_path+'test_hq/{}.jpg'.format(id))

            img = cv2.resize(img, (img_tuple[1], img_tuple[0]))
            x_batch.append(img)
        x_batch = np.array(x_batch, np.float32) / 255
        preds = model.predict_on_batch(x_batch)
        preds = np.squeeze(preds, axis=3)
        for pred in preds:
            prob = cv2.resize(pred, (orig_width, orig_height))
            mask = prob > threshold
            rle = run_length_encode(mask)
            rles.append(rle)

    print("Generating submission file...")
    df = pd.DataFrame({'img': names, 'rle_mask': rles})
    df.to_csv(output_path + model_weight.split('/')[-1] + '.csv.gz', index=False, compression='gzip')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help='Model name')
    parser.add_argument('--model_weight', help='Weight path')

    parser.add_argument('--out_path', help='Output path', default='output/')
    parser.add_argument('--in_path', help='input dir', default='input/')

    parser.add_argument('--batchsize', help='Batch size', default=24)
    parser.add_argument('--image_size', help='image size', default=256)

    parser.add_argument("--lowq", type=str2bool, nargs='?',
                        const=True, default=False,
                        help="Activate nice mode.")

    args = parser.parse_args()

    predict(args)

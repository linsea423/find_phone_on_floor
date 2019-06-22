# -*- coding: utf-8 -*-
"""
    This is used to test transfer learning on a pre-trained CNN

    Author: Linhai Li
"""

from keras import models
from keras.applications import ResNet50
from keras.applications import VGG19
from keras.applications.imagenet_utils import preprocess_input
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
from keras.layers import Flatten, Dense, Dropout
from keras.optimizers import SGD, Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, ReduceLROnPlateau, EarlyStopping
import numpy as np
import pandas as pd
import argparse
import sys
import os

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", default='find_phone', help="path to the input image")
ap.add_argument("-model", "--model", type=str, default="VGG19", help="name of pre-trained network to use. VGG19 or ResNet50.")
args = vars(ap.parse_args())

if args['model'] not in ['VGG19', 'ResNet50']:
    sys.exit('choose a model from VGG19 and ResNet50')
else:
    network = args['model']

path = args['image']

#Define hyper-parameters
NNODES = 10
IMG_ROW = 326
IMG_COL = 490
BATCH_SIZE = 32
EPOCH = 100

#Load and preprocess the images
def load_images(imgIDs):
    """
        Purpose: load images and save to an array
        Parameters:
            imgIDs: list of image ids
    """
    images = np.zeros((len(imgIDs),IMG_ROW,IMG_COL,3))
    k = 0
    for fn in imgIDs:
        filename = os.path.join(path, fn)
        im = load_img(filename, target_size=(IMG_ROW, IMG_COL))
        im = img_to_array(im)
        im = np.expand_dims(im, axis=0)
        im = preprocess_input(im)
        images[k] = im
        k += 1

    return images

#predict the centers of the phones using the trained model
def predict_center(model, imgIDs):
    """
        Purpose: predict centers of phones for the given image ids
        Parameters:
            model: trained CNN model
            imgIDs: list of image ids
    """
    images = load_images(imgIDs)
    images = images/255.

    centers = model.predict(images, batch_size=BATCH_SIZE)

    return centers

if __name__ == "__main__":
    #get the center of phone for each phone
    locs = pd.read_csv(os.path.join(path, 'labels.txt'), sep='\s+', header=None, names=['imgID', 'x', 'y'])

    #Hold out datasets to test the final performance on getting the center of phone
    test_set = locs.sample(n=8, random_state=123).reset_index(drop=True)
    locs = locs.drop(test_set.index).reset_index(drop=True)

    #Get list of imgIDs
    img_fns = locs['imgID'].values
    targets = locs[['x', 'y']]

    images = load_images(img_fns)
    images = images/255.

    #Load the pre-trained CNN model on ImageNet
    if network=='VGG19':
        base_model = VGG19(weights='vgg19_imagenet_weights_tf_dim_ordering_notop.h5', include_top=False, input_shape=(326, 490, 3))
    else:
        base_model = ResNet50(weights='resnet50_imagenet_weights_tf_dim_ordering_notop.h5', include_top=False, input_shape=(326, 490, 3))

    model = models.Sequential()
    model.add(base_model)
    model.add(Flatten())
    model.add(Dense(NNODES, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(2, activation='relu'))

    model.summary()
    base_model.trainable = False

    # model.compile(optimizer=SGD(lr=0.01), loss='mse', metrics=['mse'])
    model.compile(optimizer=Adam(lr=1.0e-6, decay=1.0e-3), loss='mse', metrics=['mse'])

    #save the weights at the checkpoints
    model_fn = '{val_loss:.4f}-loss_{epoch}epoch_'+network+'.hdf5'
    ckpt = ModelCheckpoint(filepath=model_fn, monitor='val_loss', save_best_only=True)
    #reduce the lr when the val_loss is not improved after 3 epochs
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-8)
    earlyStop = EarlyStopping(monitor='val_loss', min_delta=5.0e-4, patience=10, verbose=0, mode='auto')
    model_hist = model.fit(images, targets, batch_size=BATCH_SIZE, epochs=EPOCH, verbose=1, shuffle=True,
                           validation_split=0.2, callbacks=[ckpt, reduce_lr, earlyStop])

    # serialize model to JSON
    model_json = model.to_json()
    with open("phone_model_{}.json".format(network), "w") as json_file:
        json_file.write(model_json)

    #Check the holdout test dataset
    pred_centers = predict_center(model, test_set['imgID'].values)
    true_centers = test_set[['x', 'y']].values

    n_miss = 0
    for i in range(len(true_centers)):
        distance = np.sqrt(np.sum((pred_centers[i] - true_centers[i])**2))
        if distance>0.05:
            n_miss += 1

    print('{:d} out of 8 images were missed by {}'.format(n_miss, network))













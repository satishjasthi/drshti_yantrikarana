"""
Reference: 
Usage:

About: To create augmented data for numpy arrays using CPU before training

Author: Satish Jasthi
"""
import logging
import time

import tables
import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.python import keras

from drshti_yantrikarana.config import (OfflineDataAugImages, OfflineDataAugLabels,
                                        resize_shape, channel_depth, num_classes)
from drshti_yantrikarana.src.data.augmentation.saptialTrasformations import pad_image, random_crop, random_flip, \
    equalize, autocontrast, color, brightness


logger = tf.get_logger()
logger.setLevel(logging.DEBUG)
cifar_mu, cifar_sigma = np.array([0.4914, 0.4822, 0.4465]), np.array([0.2023, 0.1994, 0.2010])


def createAugmentedData(x_train:np.array=None,
                        y_train:np.array=None)->None:
    """
    Function to create and save augmented data on training images numpy array using following data augmentations
    - padding(4,4) and random crop (32,32)
    - horizontal flip
    - equalize
    - autoContrast
    - color
    - brightness

    """
    y_train = keras.utils.to_categorical(y_train, num_classes=num_classes)
    augmetedImageData = tables.open_file(OfflineDataAugImages, mode='w')
    augmetedLabelData = tables.open_file(OfflineDataAugLabels, mode='w')
    
    images_earray = augmetedImageData.create_earray(where=augmetedImageData.root,
                                                 name='images',
                                                 atom=tables.UInt8Atom(),
                                                 shape=[0, resize_shape[0], resize_shape[1], channel_depth])
    labels_earray = augmetedLabelData.create_earray(where=augmetedLabelData.root,
                                                 name='labels',
                                                 atom=tables.UInt8Atom(),
                                                 shape=[0, y_train.shape[1]])


    logger.info('Creating offline data augmentations..................................................................')
    logger.info(f'Number of original training images: {x_train.shape[0]}')
    for image, label in zip(x_train, y_train):

        # normalize image
        image = image/255

        # standardize image
        image - image - cifar_mu/ cifar_sigma

        # original
        images_earray.append(np.array(image)[None])
        labels_earray.append(label[None])

        # padding
        padded_image = pad_image(image=image,
                                 padding=[[4, 4], [4, 4], [0, 0]])

        # cropping
        images_earray.append(random_crop(image=padded_image,
                                 height=32,
                                 width=32,
                                 depth=3)[None])
        labels_earray.append(label[None])

        # horizontal flip
        images_earray.append(random_flip(image=image,
                                flip_mode='h')[None])
        labels_earray.append(label[None])

        # equalize
        images_earray.append(equalize(image=image)[None])
        labels_earray.append(label[None])

        # autocontrast
        images_earray.append(autocontrast(image=image)[None])
        labels_earray.append(label[None])

        # color
        images_earray.append(color(image=image)[None])
        labels_earray.append(label[None])

        # brightness
        images_earray.append(brightness(image=image)[None])
        labels_earray.append(label[None])
    logger.info(f'Number of augmented training images: {images_earray.shape[0]}')
    logger.info('Saving augmented data.................................................................................')
    augmetedImageData.close()
    augmetedLabelData.close()

if __name__ == '__main__':
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
    createAugmentedData(x_train, y_train)
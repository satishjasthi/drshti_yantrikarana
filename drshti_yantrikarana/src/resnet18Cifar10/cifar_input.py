"""
Reference: 
Usage:

About:

Author: Satish Jasthi
"""

import tables
import tensorflow as tf
from tensorflow.python import keras

from drshti_yantrikarana.config import OfflineDataAugImages, OfflineDataAugLabels, batch_size


def build_input(mode):
    if mode == 'train':
        # read hdf5 files
        x_train_data = tables.open_file(OfflineDataAugImages, mode='r')
        y_train_data = tables.open_file(OfflineDataAugLabels, mode='r')

        x_train = x_train_data.root.images
        y_train = y_train_data.root.labels
        train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))

        BATCH_SIZE = batch_size
        SHUFFLE_BUFFER_SIZE = 100

        train_dataset = train_dataset.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
        return train_dataset
    elif mode == 'test':
        _, (x_test, y_test) = keras.datasets.cifar10.load_data()
        test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
        test_dataset = test_dataset.batch(batch_size)
        return test_dataset
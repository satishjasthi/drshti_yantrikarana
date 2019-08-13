"""
Reference: 
Usage:

About:

Author: Satish Jasthi
"""

import tables

from drshti_yantrikarana.config import OfflineDataAugImages, OfflineDataAugLabels


def build_input(dataset, data_path, batch_size, mode):
    # read hdf5 files
    x_train_data = tables.open_file(OfflineDataAugImages, mode='r')
    y_train_data = tables.open_file(OfflineDataAugLabels, mode='r')

    x_train = x_train_data.root.images
    y_train = y_train_data.root.labels
    return images, labels
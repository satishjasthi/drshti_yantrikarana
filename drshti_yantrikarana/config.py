"""
Script to create a mode config dictionary which provides access to all changeable
parameters of model training pipe line

author: Satish Jasthi
------
"""
from pathlib import Path

import numpy as np
from tensorflow.python import keras

# get predefined numpy array based data here
from Utils.dataAugmentations import random_rotate_90, random_flip

(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
x_train, y_train = x_train[:20,:,:,:], y_train[:20]
x_test, y_test = x_test[:10,:,:,:], y_test[:10]

dataSetName = 'cifar10'
# dataSetName = 'Fruits'

batchSize=1

data_root_dir = Path('/Users/satishjasthi/Documents/Professional/ML/drshti_yantrikarana/data/')

data_aug_dir = data_root_dir.parent/'AugmentedData'
data_aug_dir.mkdir(parents=True, exist_ok=True)

HDF5_data_dir = data_root_dir.parent/'HDF5Data'
HDF5_data_dir.mkdir(parents=True, exist_ok=True)

Tfrecords_data_dir = data_root_dir.parent/'Tfrecords'
Tfrecords_data_dir.mkdir(parents=True, exist_ok=True)

model_config = {'name':'DavidNet',
                'dataSetName':dataSetName,
                'numClasses':10,


                'dataFormat':"numpy", # can be 'numpy' or 'image'
                'dataSource':((x_train, y_train), (x_test, y_test)), # will be tuple if numpy data
                # else Path object to the dir with train and test dir

                'img_height':32,
                'img_width':32,
                'img_depth':3,


                'train_data_dir':data_root_dir/'train',
                'test_data_dir':data_root_dir/'test',


                'augment_data_switch':False, # bool to turn on or off data augmentation
                'train_data_aug':data_aug_dir,
                'train_data_aug_hdf5': data_aug_dir/f'{dataSetName}_train_augmented.h5',
                'data_augmentation_functions':[random_rotate_90, random_flip],
                'data_augmentation_fraction': 0.5,


                'hdf5_train_data':HDF5_data_dir/f'{dataSetName}_train.h5',
                'hdf5_test_data':HDF5_data_dir/f'{dataSetName}_test.h5',

                'train_tfrecords':Tfrecords_data_dir/f'{dataSetName}_train.tfrecords',
                'test_tfrecords':Tfrecords_data_dir/f'{dataSetName}_test.tfrecords',

                'tfDatasetShuffleBool':True,
                'tfDatasetPrefetchBool':True,
                'prefetchBufferSize': batchSize,


                'loss': keras.losses.categorical_crossentropy,
                'metrics':['accuracy'],
                'optimizer':keras.optimizers.SGD(lr=0,
                                                 momentum=0.9,
                                                 decay=5e-4*512),
                'batchSize':batchSize,
                'epochs':24,
                'transition_epoch':5,
                'maxLr':0.8,
                'minLr':0.08,

                }
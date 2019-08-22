"""
Script to create a mode config dictionary which provides access to all changeable
parameters of model training pipe line

author: Satish Jasthi
------
"""
from pathlib import Path

from tensorflow.python import keras

# get predefined numpy array based data here
(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

data_root_dir = Path('Data')
data_root_dir.mkdir(parents=True, exist_ok=True)

data_aug_dir = Path('AugmentedData')
data_aug_dir.mkdir(parents=True, exist_ok=True)

HDF5_data_dir = Path('HDF5Data')
HDF5_data_dir.mkdir(parents=True, exist_ok=True)

model_config = {'name':'DavidNet',
                'dataSetName':'cifar10',
                'numClasses':10,


                'dataFormat':'numpy', # can be 'numpy' or 'image'
                'dataSource':((x_train, y_train), (x_test, y_test)), # will be tuple if numpy data
                # else Path object to the dir with train and test dir

                'img_height':32,
                'img_width':32,
                'img_depth':3,


                'train_data_dir':data_root_dir/'train',
                'test_data_dir':data_root_dir/'test',


                'augment_data_switch':False, # bool to turn on or off data augmentation
                'train_data_aug':data_aug_dir,


                'hdf5_train_data':HDF5_data_dir/'train.h5',
                'hdf5_test_data':HDF5_data_dir/'test.h5',

                'loss': keras.losses.categorical_crossentropy,
                'optimizer':keras.optimizers.SGD(lr=0,
                                                 momentum=0.9,
                                                 decay=5e-4*512),
                'batchSize':512,
                'epochs':24,
                'transition_epoch':5,
                'maxLr':0.8,
                'minLr':0.08,

                }
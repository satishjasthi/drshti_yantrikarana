"""
About: Configuration file for whole project

Author: Satish Jasthi
"""

# database configs
import logging
from pathlib import Path

db_username = 'root'
db_password = 'password'
db_name = 'drshit_yantrikarana'

#logger#################################################################################################################
LOGGER_LEVEL = logging.DEBUG

# data configs##########################################################################################################
external_data_dir = "/Volumes/SatishJ/ML/Datasets/FruitsMini"
data_dir = Path('/Users/satishjasthi/Documents/Professional/ML/drshti_yantrikarana/data')
TrainHdf5_data = data_dir.joinpath(data_dir.parent, 'HDF5_data_files/TrainData.h5')
TrainTfRecord_data = data_dir.joinpath(data_dir.parent, 'TFRecords_data_files/TrainData.tfrecords')
TestHdf5_data = data_dir.joinpath(data_dir.parent, 'HDF5_data_files/TestData.h5')
TestTfRecord_data = data_dir.joinpath(data_dir.parent, 'TFRecords_data_files/TestData.tfrecords')
OfflineDataAugImages = data_dir.joinpath(data_dir.parent, 'HDF5_data_files/AugmentedImageData.h5')
OfflineDataAugLabels = data_dir.joinpath(data_dir.parent, 'HDF5_data_files/AugmentedLabelData.h5')

# data preprocessing####################################################################################################
resize_shape = (32,32)
num_classes = 10
# mu and sigma must be tf.float64
# TODO add code to calculate data_mu and data_sigma
data_mu = ''
data_sigma = ''
channel_depth = 3
batch_size = 2

# data augmentation#####################################################################################################
# for random rotation
rotation_min = 10
rotation_max = 45

# model training########################################################################################################
model_optimizer='sgd'
model_loss='categorical_crossentropy'
model_metrics=['accuracy']
model_steps_per_epoch=None
model_epochs=100
train_verbose=1
train_callbacks=None
train_validation_steps=None
train_validation_freq=1
train_class_weight=None
train_max_queue_size=10
train_workers=1
train_use_multiprocessing=False
train_shuffle=True
train_initial_epoch=0

# datagen
datagen_featurewise_center=False,
datagen_samplewise_center=False,
datagen_featurewise_std_normalization=False,
datagen_samplewise_std_normalization=False,
datagen_zca_whitening=False,
datagen_zca_epsilon=1e-06,
datagen_rotation_range=0,
datagen_width_shift_range=0.0,
datagen_height_shift_range=0.0,
datagen_brightness_range=None,
datagen_shear_range=0.0,
datagen_zoom_range=0.0,
datagen_channel_shift_range=0.0,
datagen_fill_mode='nearest',
datagen_cval=0.0,
datagen_horizontal_flip=False,
datagen_vertical_flip=False,
datagen_rescale=None,
datagen_preprocessing_function=None,
datagen_data_format=None,
datagen_validation_split=0.0,
datagen_datagen_dtype=None

datagen_SHUFFLE_BUFFER=128
datagen_BATCH_SIZE=128

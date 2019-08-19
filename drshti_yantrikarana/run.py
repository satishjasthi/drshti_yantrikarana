"""
Reference: 
Usage:

About: Script to run a DL network from data preprocessing to evaluating

Author: Satish Jasthi
"""

import logging
import numpy as np
from pathlib import Path

import tensorflow as tf

from data import ConvertData2Hdf5, TFRecords
from moduleLogger import DyLogger

logger = DyLogger(logging_level=logging.DEBUG)

class DrshtiYantrikarana(object):

    def __init__(self,
                 modelName: str = None,
                 raw_data: Path = None,
                 ):
        self.modelName = modelName
        self.logger = logger.get_logger(__name__)
        self.raw_data = raw_data
        self.trainTestSplitDir = self.raw_data / 'TrainTestSplitData'

        self.hdf5_save_dir = self.raw_data / 'HDF5Data'
        self.TrainHdf5_data = self.hdf5_save_dir/'train.h5'
        self.TestHdf5_data = self.hdf5_save_dir / 'test.h5'

        self.tfrecords_save_dir = self.raw_data / 'TFRecords'
        self.TrainTfRecord_data = self.tfrecords_save_dir/'train'
        self.TestTfRecord_data = self.tfrecords_save_dir / 'test'


    def prepareModelData(self, data_format: (np.array, str) = None,
                         x_train: np.array = None,
                         y_train: np.array = None,
                         x_test: np.array = None,
                         y_test: np.array = None,
                         resizeHeight: int = None,
                         resizeWidth: int = None,
                         augment_bool: bool = None,
                         augmentations_list: list = None,
                         save_augmentation_flag: bool = None

                         ):
        self.logger.logging.info(
            'Preparing data for model training..........................................................................')

        # read raw data and create hdf5 files
        self.logger.logging.debug(
            f'Raw data is in {data_format} format.................................................')
        self.logger.logging.info(
            'Converting raw data to hdf5 files.....................................................')
        self.hdf5DataCreator = ConvertData2Hdf5(data_dir=self.trainTestSplitDir,
                                                hdf5_save_dir=self.hdf5_save_dir,
                                                x_train_arr=x_train,
                                                y_train_arr=y_train,
                                                x_test_arr=x_test,
                                                y_test_arr=y_test,
                                                data_format=data_format,
                                                resizeHeight=resizeHeight,
                                                resizeWidth=resizeWidth,
                                                augment_bool=augment_bool,
                                                augmentations_list=augmentations_list,
                                                save_augmentation_flag=save_augmentation_flag)
        self.logger.logging.debug(
            'Finished converting raw data to hdf5 files.....................................................')

        # create tfrecords from hdf5 files
        self.logger.logging.info(
            'Converting hdf5 data to TFRecords..........................................................................')

        self.tfRecordCreator = TFRecords(mode='train',
                                         TrainHdf5_data=self.TrainHdf5_data,
                                         TestHdf5_data=self.TestHdf5_data,
                                         TrainTfRecord_data=self.TrainTfRecord_data,
                                         TestTfRecord_data=self.TrainTfRecord_data)
        self.tfRecordCreator.writeTfRecord()
        self.parsed_image_dataset = self.tfRecordCreator.writeTfRecord()

        self.dataset = tf.data.Dataset.from_generator(self.parsed_image_dataset)

        # create tf dataset from tf records

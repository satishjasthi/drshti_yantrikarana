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
from tensorflow.python import keras

from data import ConvertData2Hdf5, TFRecords, preprocessTfDataset
from models import DavidNet
from moduleLogger import DyLogger
from test import EvaluateModel
from train import ModelTraining


logger = DyLogger(logging_level=logging.DEBUG).logger

class DrshtiYantrikarana(object):

    def __init__(self,
                 modelName: str = None,
                 num_classes=None,
                 raw_data: Path = None,
                 data_mean: list = None,
                 data_std: list = None
                 ):
        self.modelName = modelName
        self.num_classes = num_classes
        self.raw_data = raw_data
        self.trainTestSplitDir = self.raw_data / 'TrainTestSplitData'

        self.hdf5_save_dir = self.raw_data / 'HDF5Data'
        self.TrainHdf5_data = self.hdf5_save_dir/'train.h5'
        self.TestHdf5_data = self.hdf5_save_dir / 'test.h5'

        self.tfrecords_save_dir = self.raw_data / 'TFRecords'
        self.tfrecords_save_dir.mkdir(exist_ok=True, parents=True)
        self.TrainTfRecord_data = self.tfrecords_save_dir/'train.tfrecords'
        self.TestTfRecord_data = self.tfrecords_save_dir / 'test.tfrecords'


    def prepareModelData(self, data_format: (np.array, str) = None,
                         x_train: np.array = None,
                         y_train: np.array = None,
                         x_test: np.array = None,
                         y_test: np.array = None,
                         resizeHeight: int = None,
                         resizeWidth: int = None,
                         augment_bool: bool = None,
                         augmentations_list: list = None,
                         save_augmentation_flag: bool = None,
                         batch_size = None,
                         )->tuple:
        logger.info(
            f'Preparing data for model training..........................................................................')

        # read raw data and create hdf5 files
        logger.debug(
            f'Raw data is in {data_format} format.................................................')
        logger.info(
            'Converting raw data to hdf5 files.....................................................')
        self.hdf5DataCreator = ConvertData2Hdf5(data_dir=self.trainTestSplitDir,
                                                num_classes=self.num_classes,
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
        self.hdf5DataCreator.createTrainTestHdf5Files()
        logger.debug(
            'Finished converting raw data to hdf5 files.....................................................')

        # create tfrecords from hdf5 files
        logger.info(
            'Converting hdf5 data to TFRecords..........................................................................')

        self.tfRecordCreator_train = TFRecords(mode='train',
                                               num_classes=self.num_classes,
                                               TrainHdf5_data=self.TrainHdf5_data,
                                               TestHdf5_data=self.TestHdf5_data,
                                               TrainTfRecord_data=self.TrainTfRecord_data,
                                               TestTfRecord_data=self.TestTfRecord_data)
        self.tfRecordCreator_test = TFRecords(mode='test',
                                               num_classes=self.num_classes,
                                               TrainHdf5_data=self.TrainHdf5_data,
                                               TestHdf5_data=self.TestHdf5_data,
                                               TrainTfRecord_data=self.TrainTfRecord_data,
                                               TestTfRecord_data=self.TestTfRecord_data)
        self.tfRecordCreator_train.writeTfRecord()
        self.tfRecordCreator_test.writeTfRecord()
        train_dataset = self.tfRecordCreator_train.readTfRecord()
        test_dataset = self.tfRecordCreator_test.readTfRecord()

        # define batch size and apply prefetch
        train_dataset = train_dataset.shuffle(buffer_size=batch_size*2)
        # train_dataset = train_dataset.map(map_func=preprocessTfDataset)
        train_dataset = train_dataset.batch(batch_size=batch_size)
        train_dataset = train_dataset.prefetch(buffer_size=batch_size*2)

        test_dataset = test_dataset.shuffle(buffer_size=batch_size * 2)
        # test_dataset = test_dataset.map(map_func=preprocessTfDataset)
        test_dataset = test_dataset.batch(batch_size=batch_size)
        test_dataset = test_dataset.prefetch(buffer_size=batch_size * 2)
        return train_dataset, test_dataset

    def train_model(self,
                    model_name: str,
                    train_dataset:tf.data.Dataset,
                    test_dataset:tf.data.Dataset,
                    loss:keras.losses,
                    epochs:int,
                    batch_size:int,
                    optimizer:keras.optimizers,
                    metrics:list=['accuracy'],
                    )->ModelTraining:

        # define network in keras
        network = DavidNet()
        kmodel = network.get_model()

        # define callbacks
        callbacks_list = []
        name = 'custom'

        # define model trainer train
        modelTrainer = ModelTraining(kmodel = kmodel,
                                     loss = loss,
                                     epochs = epochs,
                                     batch_size = batch_size,
                                     optimizer = optimizer,
                                     metrics = metrics,
                                     train_dataset = train_dataset,
                                     test_dataset = test_dataset,
                                     name = name)

        # compile and train model
        modelTrainer.compileModel()
        modelTrainer.trainModel()

        return modelTrainer

    def evaluate_model(self, modelTrainer:ModelTraining, x_test:np.array):
        model_evaluator = EvaluateModel()
        pass
    

def main():
    # cntr = DrshtiYantrikarana(raw_data=Path(r'C:\Users\neere\Desktop\deleteme\raw_dir'), num_classes=10)
    cntr = DrshtiYantrikarana(raw_data=Path('../'), num_classes=10)
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
    # x_train  = x_train[0:100, :, :, :]
    # y_train = y_train[0:100,:]
    # x_test = x_test[0:100, :,:,:]
    # y_test = y_test[0:100,:]

    train_dataset, test_dataset = cntr.prepareModelData(data_format="np_array",
                                                          x_train=x_train,
                                                          x_test=x_test,
                                                          y_train=y_train,
                                                          y_test=y_test,
                                                          batch_size=512,
                                                          resizeHeight=32,
                                                          resizeWidth=32,
                                                          augment_bool=False,
                                                          augmentations_list=['random_rotate', 'horizonatal_flip'],
                                                          save_augmentation_flag=False)
    cntr.train_model(model_name='custom',
                     train_dataset=train_dataset,
                     test_dataset=test_dataset,
                     batch_size=512,
                     loss=keras.losses.categorical_crossentropy,
                     epochs=24,
                     optimizer=keras.optimizers.sgd(lr=0, momentum=0.9, decay=5e-4*512, nesterov=True))

if __name__ == "__main__":
    main()

    # TODO Data
        # TODO create raw data backup
        # TODO Class specific DA

    # TODO Create metrics model by adding
        #  Gradcam
        #  miss classified images
        #  Image gallery of raw and aug images
        #  Image gallery of missclassified images
        # TODO Funciton to find MaxLr
        # TODO Function to select LR strategy
        # TODO Add timer
        # TODO Add Central logger

    # o = DrshtiYantrikarana(modelName='ResNet', raw_data=Path(r"/Users/satishjasthi/Downloads/deleteme/raw_dir"))
    # o.prepareModelData(data_format="images",resizeWidth=90, resizeHeight=90, augment_bool=True,  save_augmentation_flag=True, augmentations_list=['random_rotate', 'horizonatal_flip' ], batch_size=1)

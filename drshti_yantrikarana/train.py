"""
Reference: 
Usage:

About: Class to train a model

Author: Satish Jasthi
"""
from multiprocessing import cpu_count


import tensorflow as tf
from tensorflow.python import keras
import matplotlib.pyplot as plt

class ModelTraining(object):

    """
    Class to train a model defined in keras
    """

    def __init__(self,
                 kmodel:keras.models.Model=None,
                 train_dataset:tf.data.Dataset=None,
                 test_dataset:tf.data.Dataset=None,
                 loss:str=None,
                 epochs:int=None,
                 batch_size:int=None,
                 optimizer:keras.optimizers=None,
                 metrics:list=None,
                 callbacks_list:list=None,
                 name:str=None
                 ):
        self.kmodel = kmodel
        self.loss = loss
        self.epochs = epochs
        self.batch_size = batch_size
        self.optimizer = optimizer
        self.metrics = metrics
        self.callbacks_list = callbacks_list
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.name = name

    def compileModel(self):
        self.kmodel.compile(loss=self.loss,
                            optimizer=self.optimizer,
                            metrics=self.metrics
                            )

    def save_accuracyLoss_plots(self):
        """
        Method to save train and test accuracy and loss values
        :return:
        """
        exp = "Model:{}_epochs:{}".format(self.name, self.epochs)
        # summarize history for accuracy
        plt.plot(self.model_history.history['acc'])
        plt.plot(self.model_history.history['val_acc'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.savefig(f'{exp}_accuracy.png')

        # summarize history for loss
        plt.plot(self.model_history.history['loss'])
        plt.plot(self.model_history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.savefig(f'{exp}_loss.png')

    def trainModel(self):
        """
        Method to train model
        :return:
        """
        # compile model
        self.compileModel()

        #train model
        self.model_history = self.kmodel.fit(self.train_dataset,
                                             epochs=self.epochs,
                                             callbacks=self.callbacks_list,
                                             validation_data=self.test_dataset,
                                             workers=cpu_count()-3
                                             )


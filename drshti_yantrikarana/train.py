"""
Reference: 
Usage:

About: Class to train a model

Author: Satish Jasthi
"""
import datetime
from multiprocessing import cpu_count
from pathlib import Path

import tensorflow as tf
from tensorflow.python import keras
import matplotlib.pyplot as plt
from tf_explain.callbacks import GradCAMCallback, OcclusionSensitivityCallback, ActivationsVisualizationCallback

from utils import OneCycleLrScheduler


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
                 name:str=None,
                 enable_gradcam: bool = None,
                 enable_activation_map: bool = None,
                 enable_occlusion_sensitivity: bool = None,
                 activation_map_layer: str = None,
                 grad_cam_activation_layer: str = None,
                 grad_cam_output_dir:str=None,
                 occlusion_output_dir:str=None,
                 activation_map_output_dir:str=None,
                 num_classes:int=None,
                 ):
        self.kmodel = kmodel
        self.loss = loss
        self.epochs = epochs
        self.batch_size = batch_size
        self.optimizer = optimizer
        self.metrics = metrics
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.num_classes=num_classes
        self.name = name
        self.tflogdir = f"{self.name}_tb_logs/scalars" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.enable_gradcam=enable_gradcam
        self.enable_activation_map=enable_activation_map
        self.enable_occlusion_sensitivity=enable_occlusion_sensitivity
        self.activation_map_layer=activation_map_layer
        self.grad_cam_activation_layer=grad_cam_activation_layer
        self.grad_cam_output_dir=grad_cam_output_dir
        self.occlusion_output_dir=occlusion_output_dir
        self.activation_map_output_dir=activation_map_output_dir

        # tb summaries writer
        file_writer = tf.summary.create_file_writer(self.tflogdir + '/metrics')
        file_writer.set_as_default()

    def compileModel(self):
        self.kmodel.compile(loss=self.loss,
                            optimizer=self.optimizer,
                            metrics=self.metrics
                            )
        self.tb_callbacks = keras.callbacks.TensorBoard(log_dir=self.tflogdir)

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
        # weights checkpointing
        filepath = f"{self.name}_ModelWeights/"+"weights_" +"{epoch:02d}-{val_acc:.2f}.hdf5"
        checkpoint = keras.callbacks.ModelCheckpoint(filepath,
                                                     monitor='val_acc',
                                                     verbose=1,
                                                     save_best_only=True,
                                                     mode='max')

        # callbacks
        call_backs_list = [keras.callbacks.LearningRateScheduler(OneCycleLrScheduler),
                           self.tb_callbacks,
                           checkpoint
                           ]

        if self.enable_gradcam:
            grad_cam_callbacks = []
            for index in range(self.num_classes):
                grad_cam_callbacks.append(GradCAMCallback(validation_data=self.test_dataset,
                                        layer_name=self.grad_cam_activation_layer,
                                        class_index=index,
                                        output_dir=self.grad_cam_output_dir)
                                          )
            call_backs_list.extend(grad_cam_callbacks)


        if self.enable_occlusion_sensitivity:
            occlusion_callbacks = []
            for index in range(self.num_classes):
                occlusion_callbacks.append(OcclusionSensitivityCallback(
                                            validation_data=self.test_dataset,
                                            class_index=index,
                                            patch_size=4,
                                            output_dir=self.occlusion_output_dir)
                                           )
            call_backs_list.extend(occlusion_callbacks)

        if self.enable_activation_map:
            activation_callbacks = []
            for index in range(self.num_classes):
                activation_callbacks.append(ActivationsVisualizationCallback(
                                                    validation_data=self.test_dataset,
                                                    layers_name=self.activation_map_layer,
                                                    output_dir=self.activation_map_output_dir))
            call_backs_list.extend(activation_callbacks)


        #train model
        self.model_history = self.kmodel.fit(self.train_dataset,
                                             epochs=self.epochs,
                                             callbacks=call_backs_list,
                                             validation_data=self.test_dataset,
                                             )


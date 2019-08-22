"""
Reference: 
Usage:

About:

Author: Satish Jasthi
"""
import tensorflow as tf
from tensorflow.python import keras
from typing import Callable


class OptimalTransitionLrFinder(object):

    def __init__(self,
                 model:keras.models.Model,
                 lr_schedule:Callable=None,
                 transition_epochs:list=None,
                 max_Lr:int=None,
                 min_Lr:int=None,
                 ):
        self.model=model
        self.lr_schedule=lr_schedule
        self.transition_epochs=transition_epochs
        self.max_Lr=max_Lr
        self.min_Lr=min_Lr

    def test_transition_epochs(self,
                               train_dataset:tf.data.Dataset=None,
                               test_dataset:tf.data.Dataset=None
                               ):
        pass

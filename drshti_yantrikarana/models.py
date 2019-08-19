"""
Reference: 
Usage:

About:

Author: Satish Jasthi
"""

from tensorflow.python import keras
from tensorflow.python.keras.layers import *
from tensorflow.python.keras import applications

class Networks(object):

    def __init__(self,
                 depth:int=None,
                 num_class:int=None,
                 input_shape:tuple=None):
        self.depth = depth
        self.input_shape = input_shape
        self.num_class = num_class

class Resenet(Networks):
    """
    Class to build pretrained resnet model
    """
    def __init__(self, *args, **kwargs):
        super(Resenet, self).__init__(*args, **kwargs)


    def get_pretrained_model(self):
        input_tensor = Input(shape=self.input_shape)
        base_model = applications.resnet50.ResNet50(
            input_tensor=input_tensor,
            include_top=False,
            weights='imagenet',
            backend=keras.backend,
            layers=keras.layers,
            models=keras.models,
            utils=keras.utils)

        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        if self.num_class == 2:
            actv = 'sigmoid'
        else:
            actv = 'softmax'

        predictions = Dense(units=self.num_class, activation=actv, name='predictions')(x)
        model = keras.models.Model(input_tensor, predictions)

        return model

class DenseNet(Networks):
    """
    Class to build pretrained DenseNet model
    """

    def __init__(self, *args, **kwargs):
        super(DenseNet, self).__init__(*args, **kwargs)

    def get_pretrained_model(self):
        input_tensor = Input(shape=self.input_shape)

        base_model = applications.densenet.DenseNet121(input_tensor=input_tensor,
                                                       include_top=False, weights='imagenet',
                                                       backend=keras.backend,
                                                       layers=keras.layers,
                                                       models=keras.models,
                                                       utils=keras.utils)

        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        if self.num_class == 2:
            actv = 'sigmoid'
        else:
            actv = 'softmax'

        predictions = Dense(units=self.num_class, activation=actv, name='predictions')(x)
        model = keras.models.Model(input_tensor, predictions)

        return model

if __name__ == "__main__":
    o = Resenet(input_shape=(224,224,3),
                num_class=4
                )
    model = o.get_pretrained_model()
    print(model.summary())

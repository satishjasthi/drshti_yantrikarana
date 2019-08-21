"""
Reference: 
Usage:

About:

Author: Satish Jasthi
"""

from tensorflow.python import keras
from tensorflow.python.keras.initializers import glorot_uniform
from tensorflow.python.keras.layers import *
from tensorflow.python.keras import applications

class Networks(object):

    def __init__(self,
                 depth:int=None,
                 num_class:int=None,
                 input_shape:tuple=None,
                 name=None
                 ):
        self.depth = depth
        self.input_shape = input_shape
        self.num_class = num_class
        self.name=None

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

class DavidNet(Networks):
    """
    Custom class to train mnist data
    """

    def __init__(self, *args, **kwargs):
        super(DavidNet, self).__init__(*args, **kwargs)

    def get_model(self):
        # Network definition
        def conv_bn_relu(inputLayer: keras.layers, kernel_size: int, kernel_num: int, strides, layer_name: str):
            # def conv_bn(c_in, c_out, bn_weight_init=1.0, **kw):
            # return {
            #     'conv': nn.Conv2d(c_in, c_out, kernel_size=3, stride=1, padding=1, bias=False),
            #     'bn': batch_norm(c_out, bn_weight_init=bn_weight_init, **kw),
            #     'relu': nn.ReLU(True)
            # }

            # defining name basis
            # name_base = 'layer_' + str(layername)
            name_base = layer_name
            X = Conv2D(filters=kernel_num, kernel_size=kernel_size, strides=strides, padding='same',
                       name=name_base + '_conv', kernel_initializer=glorot_uniform(seed=0), use_bias=False)(inputLayer)
            X = BatchNormalization(name=name_base + '_bn')(X)
            X = ReLU(name=name_base + '_relu')(X)

            return X

        def general_block(inputLayer: keras.layers, nonres_filters, res1_filters, res2_filters, layer_name):
            # Retrieve Filters
            # F1, F2, F3 = filters
            # Intial Conv Block with Stride=2
            X = conv_bn_relu(inputLayer, (3, 3), nonres_filters, strides=(1, 1), layer_name=layer_name)
            X = MaxPooling2D(name=layer_name + '_maxpool', pool_size=(2, 2))(X)

            # Save the above value. You'll need this later to add back to the main path.
            X_shortcut = X

            # Res blocks
            X = conv_bn_relu(X, (3, 3), res1_filters, strides=(1, 1), layer_name=layer_name + '_res1')
            X = conv_bn_relu(X, (3, 3), res2_filters, strides=(1, 1), layer_name=layer_name + '_res2')

            # Final step: Add shortcut value to main path
            X = Add()([X, X_shortcut])

            return X

        def model_builder():
            input = Input(shape=(32, 32, 3))

            # channels = channels or {'prep': 64, 'layer1': 128, 'layer2': 256, 'layer3': 512}
            f1 = 64  # No of filters in Prep Layer conv

            f2 = 128  # No of filters in Layer_1 conv
            f3 = 128  # No of filters in Layer_1 Res1 conv
            f4 = 128  # No of filters in Layer_1 Res2 conv

            f5 = 256  # No of filters in Layer_2 conv

            f6 = 512  # No of filters in Layer_3 conv
            f7 = 512  # No of filters in Layer_3 Res1 conv
            f8 = 512  # No of filters in Layer_3 Res2 conv

            X = conv_bn_relu(input, (3, 3), kernel_num=f1, strides=(1, 1), layer_name='layer_prep')  # size = 32x32
            X = general_block(X, f2, f3, f4, 'layer_1')  # size =
            X = conv_bn_relu(X, (3, 3), kernel_num=f5, strides=(1, 1), layer_name='layer_2')
            X = MaxPooling2D(name='layer_2_maxpool', pool_size=(2, 2))(X)
            X = general_block(X, f6, f7, f8, 'layer_3')

            X = MaxPooling2D(pool_size=(4, 4), name='classifer_layer_maxpool')(X)
            X = Flatten()(X)
            X = Dense(units=10)(X)
            X = Softmax()(X)

            model = keras.models.Model(inputs=input, outputs=X)
            return model
        return model_builder()


if __name__ == "__main__":
    o = Resenet(input_shape=(224,224,3),
                num_class=4,
                name="Resenet"
                )
    model = o.get_pretrained_model()
    print(model.summary())

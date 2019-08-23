"""
Reference: 
Usage:

About:

Author: NJ2020
"""
from tensorflow.python import keras
from tensorflow.python.keras.initializers import glorot_uniform
from tensorflow.python.keras.layers import *

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

def davidNet():
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
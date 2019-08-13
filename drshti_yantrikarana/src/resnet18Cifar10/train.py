"""
Reference: 
Usage:

About:

Author: Satish Jasthi
"""
from multiprocessing import cpu_count

from drshti_yantrikarana.src.resnet18Cifar10.cifar_input import build_input
from drshti_yantrikarana.src.resnet18Cifar10.resnet18 import resnet_v1

model = resnet_v1(input_shape=(32,32,3))
model.compile(loss='categorical_crossentrpoy',
              optimizer='sgd',
              metrics=['accuracy']
              )
model.fit(build_input(mode='train'), epochs=10, validation_data=build_input(mode='test'), workers=cpu_count())
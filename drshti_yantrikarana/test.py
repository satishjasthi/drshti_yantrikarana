"""
Reference: 
Usage:

About: class to evaluate keras model on
    - accuracy
    - confusion matrix
    - recall
    - precision
    And saving miss classified images
Author: Satish Jasthi
"""
import tensorflow as tf
import numpy as np
from tensorflow.python import keras
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels


def plot_confusion_matrix(y_true:np.array,
                          y_pred:np.array,
                          classes:np.array,
                          normalize:bool=False,
                          title:str=None,
                          modelName:str=None,
                          epochs:int=None,
                          cmap=plt.cm.Blues)->None:
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    plt.savefig("Model:{}_epochs:{}.png".format(modelName, epochs))

class EvaluateModel():

    def __init__(self,
                 kmodel:keras.models.Model=None,
                 x_test:tf.data.Dataset=None,
                 y_test:tf.data.Dataset=None,
                 class_index_map:dict=None,
                 model_name:str=None,
                 train_epochs:int=None
                 ):
        self.kmodel = kmodel
        self.x_test = x_test
        self.y_test = y_test
        self.class_index_map = class_index_map
        self.model_name = model_name
        self.train_epochs= train_epochs

    def test(self):
        predictions_one_hot = self.kmodel.predict(self.x_test)
        predictions = [np.argmax(prediction) for prediction in predictions_one_hot]
        self.test_accuracy = keras.metrics.accuracy(self.y_test, predictions)
        classes = self.class_index_map.keys()
        self.tn, self.fp, self.fn, self.tp = confusion_matrix(self.y_test,
                                          predictions,
                                          labels=classes)
        self.recall = self.tp/ (self.tp + self.fn)
        self.precision = self.tp/ (self.tp + self.fp)

        # save confusion matrix
        plot_confusion_matrix(y_true=self.y_test,
                              y_pred=predictions,
                              classes=np.array(classes),
                              normalize=True,
                              title=None,
                              modelName=self.model_name,
                              epochs=self.train_epochs,
                              cmap = plt.cm.Blues)

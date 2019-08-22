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
from pathlib import Path

import tensorflow as tf
import seaborn as sns
import pandas as pd
import numpy as np
from PIL import Image
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
                 test_dataset:tf.data.Dataset=None,
                 class_index_map:dict=None,
                 model_name:str=None,
                 train_epochs:int=None,
                 save_miss_classified:bool=None,
                 save_miss_classified_dir:Path=None,

                 ):
        self.kmodel = kmodel
        self.test_dataset = test_dataset
        self.class_index_map = class_index_map
        self.model_name = model_name
        self.train_epochs= train_epochs
        self.save_miss_classified=save_miss_classified
        self.save_miss_classified_dir=save_miss_classified_dir

    def miss_classified_images_plot(self,predictions:np.array=None,
                                    true_labels:np.array=None):
        """
        Method to plot bar graph of number miss classified images for each class
        """
        classwise_miss_classified_map = {class_name:0 for class_name in list(self.class_index_map.keys())}
        for pred, label in zip(predictions, true_labels):
            if pred != label:
                class_name = self.class_index_map[label]
                classwise_miss_classified_map[class_name] += 1
        count_plot_image = self.save_miss_classified_dir/f'miss_classified_count_plot.jpg'
        count_df = pd.DataFrame(data={'classes':list(classwise_miss_classified_map.keys()),
                                      'miss_classified_freq': list(classwise_miss_classified_map.values())
                                      })
        plot_image = sns.countplot(x='classes', data=count_df)
        plot_image.figure.savefig(count_plot_image)

    def save_miss_classified_images(self, predictions:np.array=None,
                                    true_labels:np.array=None,
                                    image_arrays:np.array=None,
                                    ):
        """
        Method to save miss classified images in folders as show below

        miss_classified_images_dir:
            class1_dir(ActualLabel):
                image_classN(where classN is the predicted class)

        both predictions and true_labels are integer based arrays and not one-hot encoded vectors
        """
        assert len(predictions[0].shape) < 2, "predictions array cannot be one-hot encoded vector"
        assert len(true_labels[0].shape) < 2, "true_labels array cannot be one-hot encoded vector"
        classes = self.class_index_map.keys()

        # create class wise folders
        for class_name in classes:
            class_dir = self.save_miss_classified_dir/f'{class_name}'
            class_dir.mkdir(parents=True, exist_ok=True)

        # save images in respective folders
        image_index=0
        for pred, y_true in zip(predictions, true_labels):
            if pred != y_true:
                class_name = self.class_index_map[y_true]
                predicted_name = self.class_index_map[pred]
                dest = self.save_miss_classified/f'{class_name}/{predicted_name}_{image_index}.jpg'
                image = image_arrays[image_index]
                Image.fromarray(image.astype('uint8')).save(dest)
            image_index += 1

    def test(self):
        images, labels = [], []
        for image, labels in self.test_dataset:
            images.append(image.numpy())
            labels.append(np.argmax(labels))

        predictions_one_hot = self.kmodel.predict(images)
        predictions = [np.argmax(prediction) for prediction in predictions_one_hot]

        assert len(predictions[0].shape) < 2, "predictions array cannot be one-hot encoded vector"
        assert len(labels[0].shape) < 2, "true_labels array cannot be one-hot encoded vector"

        self.test_accuracy = keras.metrics.accuracy(labels, predictions)
        classes = self.class_index_map.keys()
        self.tn, self.fp, self.fn, self.tp = confusion_matrix(labels,
                                          predictions,
                                          labels=classes)
        self.recall = self.tp/ (self.tp + self.fn)
        self.precision = self.tp/ (self.tp + self.fp)

        # save confusion matrix
        plot_confusion_matrix(y_true=labels,
                              y_pred=predictions,
                              classes=np.array(classes),
                              normalize=True,
                              title=None,
                              modelName=self.model_name,
                              epochs=self.train_epochs,
                              cmap = plt.cm.Blues)

        if self.save_miss_classified:
            self.save_miss_classified_images(predictions=predictions,
                                             true_labels=labels,
                                             image_arrays=images
                                             )


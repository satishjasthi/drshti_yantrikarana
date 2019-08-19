"""
Reference: 
Usage: Utility class to print few images from numpy arrays, denormalize images and gradcam

About:

Author: @NJ2020
"""
from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image
import numpy as np


def save_few_images_to_dir(imagesarray=None, labelsarray=None, class_names_list=None, save_dir=None):
    # imagesarray is train_features
    # labelsarray is train_labels

    num_classes = len(np.unique(labelsarray))
    class_ids = np.unique(labelsarray)  # Note this gives only class ids i.e. 0, 1, 2.....9
    class_names_list = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    fig = plt.figure(figsize=(8, 3))

    file_ext = '.jpg'

    for i in range(num_classes):
        ax = fig.add_subplot(2, 5, 1 + i, xticks=[], yticks=[])
        idx = np.where(labelsarray[:] == i)[0]
        # Note shape of idx is (5000,)
        # np.where returns the index (not value) of all the locations where i is matching
        # Shape of labelsarray is (50000,). Total 10 classes. Number of images for each class is 5000
        # That means location indexes   of 5000 images of each class in the labelsarray for which class id is i

        features_idx = imagesarray[idx,
                       ::]  # Shape of features_idx is (5000, 32, 32, 3). Getting all the images corresponsing to idx (5000,)
        # print ('features_idx' + str(features_idx.shape))

        img_num = np.random.randint(features_idx.shape[0])  # img_num is a single value e.g. 537 or 2950 etc.
        # print ("img_num", str(img_num))

        im = features_idx[img_num]  # Getting the actual image corresponding to index = img_num
        ax.set_title(class_names_list[i])
        #plt.imshow(im)

        # Saving the images to disk
        pil_img = Image.fromarray(im.astype('uint8'))
        dest = save_dir / f'{class_names_list[i]}_{img_num}{file_ext}'
        pil_img.save(dest.as_posix())
    #
    # plt.show()


if __name__ == '__main__':
    import tempCIFAR

    # Load the raw CIFAR-10 data
    cifar10_dir = 'D:/PyCharmProjects/cifar10-fast/data/cifar-10-batches-py'
    X_train, y_train, X_test, y_test = tempCIFAR.load_CIFAR10(cifar10_dir)

    save_dir = Path('C:/Users/neere/Desktop/deleteme/temp')
    save_few_images_to_dir(imagesarray=X_train, labelsarray=y_train, class_names_list=None, save_dir=save_dir)

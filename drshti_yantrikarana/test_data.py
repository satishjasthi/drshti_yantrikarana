"""
Reference: 
Usage:

About:

Author: Satish Jasthi
"""
from unittest import TestCase

import tables

from Utils.dataUtils import readTfRecord
from yantrikarana import Data


class TestData(TestCase):
    def test_read_image_data(self):
        # create data class
        data = Data()

        # create train and test hdf5 files from images
        data.read_image_data()

        # num of arrays in hdf5 data
        train_hdf5 = tables.open_file(data.hdf5_train_data, mode='r')
        test_hdf5 = tables.open_file(data.hdf5_test_data, mode='r')

        # num of images in train and test dir
        num_train_images = len(list(data.train_data_dir.glob('*/*')))
        num_test_images = len(list(data.test_data_dir.glob('*/*')))

        num_train_img_arrs = train_hdf5.root.images.shape[0]
        num_test_img_arrs = test_hdf5.root.images.shape[0]

        self.assertEqual(num_train_images == num_train_img_arrs)
        self.assertEqual(num_test_images == num_test_img_arrs)

    def test_read_raw_data(self):
        data = Data()

        # create train and test hdf5 files from numpy or image data
        data.read_raw_data()

        # check for number of samples match if source is numpy
        if data.dataFormat == "numpy":
            train_data, test_data = data.read_raw_data()

            # load hdf5 files
            train = tables.open_file(train_data, mode='r')
            test = tables.open_file(test_data, mode='r')

            # get numbr of samples
            train_images_arrs = train.root.images.shape[0]
            train_labels_arrs = train.root.labels.shape[0]
            test_images_arrs = test.root.images.shape[0]
            test_labels_arrs = test.root.labels.shape[0]

            # num of images in train and test np array
            num_train_images = data.dataSource[0][0].shape[0]
            num_test_images = data.dataSource[1][0].shape[0]

            self.assertEqual(num_train_images, train_images_arrs)
            self.assertEqual(train_images_arrs, train_labels_arrs)
            self.assertEqual(num_test_images, test_images_arrs)
            self.assertEqual(test_images_arrs, test_labels_arrs)

        # TODO write test case for image source

    def test_augment_data(self):
        data = Data()

        # create hdf5 file of augmented data
        data.augment_data(data=Path(data.train_data_aug_hdf5))

        # read hdf5 file
        aug_data = tables.open_file(data.train_data_aug_hdf5, mode='r')

        # check number of augmentations
        num_aug_imgs = data.total_aug_images
        num_aug_used = len(data.data_augmentation_functions)

        # for numpy
        if data.dataFormat == 'numpy':
            actual_num_images = data.dataSource[0][0].shape[0]

        # for images
        if data.dataFormat == 'image':
            actual_num_images = len(list(data.train_data_dir.glob('*/*')))

        # check whether aug imgs is equal to desired number of augs
        self.assertEqual(num_aug_imgs, int(round(actual_num_images * data.data_augmentation_fraction * num_aug_used)))

    def test_createTfrecords(self):
        data = Data()

        # create train and test tf records
        train_records, test_records = data.createTfrecords()

        # read tf records
        train_dataset = readTfRecord(data.train_tfrecords)
        test_dataset = readTfRecord(data.test_tfrecords)

        # for numpy
        if data.dataFormat == 'numpy':
            train_actual_num_images = data.dataSource[0][0].shape[0]
            test_actual_num_images = data.dataSource[1][0].shape[0]

        # for images
        if data.dataFormat == 'image':
            train_actual_num_images = len(list(data.train_data_dir.glob('*/*')))
            test_actual_num_images = len(list(data.test_data_dir.glob('*/*')))

        tf_train_images, tf_test_images = 0, 0

        # calculate num of images in test and train record files
        for image, label in train_dataset:
            tf_train_images += 1
            self.assertEqual(len(label.numpy().shape), 1)
            self.assertEqual(label.numpy().shape[-1], data.numClasses)
        self.assertEqual(tf_train_images, train_actual_num_images)

        for image, label in test_dataset:
            tf_test_images += 1
            self.assertEqual(len(label.numpy().shape), 1)
            self.assertEqual(label.numpy().shape[-1], data.numClasses)
        self.assertEqual(tf_test_images, test_actual_num_images)

    def test_Tfrecords2TfDatasets(self):
        data = Data()

        # create train and test datasets
        trainDataset, testDataset = data.createTfdatasets()

        train_data = tables.open_file(data.hdf5_train_data, mode='r')
        test_data = tables.open_file(data.hdf5_test_data, mode='r')

        num_train_batches = int(round(train_data.root.images.shape[0]/data.batchSize))
        index = 0
        for image, label in trainDataset:
            index += 1
            if index != num_train_batches - 1:
                self.assertEqual(image.numpy().shape[0], data.batchSize)
                self.assertEqual(label.numpy().shape[-1], data.numClasses)
        self.assertEqual(num_train_batches, index)

        num_test_batches = int(round(test_data.root.images.shape[0] / data.batchSize))
        index = 0
        for image, label in testDataset:
            index += 1
            if index != num_test_batches - 1:
                self.assertEqual(image.numpy().shape[0], data.batchSize)
                self.assertEqual(label.numpy().shape[-1], data.numClasses)
        self.assertEqual(num_test_batches, index)

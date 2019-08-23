"""
Script to 

author: Satish Jasthi
------
"""
import sys
from pathlib import Path

import numpy as np

# add current file to sys paths
import tables

from Utils.dataUtils import Images2HDF5File, map_augmentation, numpy2hdf5, HDF52Tfrecords, Tfrecords2TfDatasets

current_file_abs_path = Path(__file__).resolve()
sys.path.append(current_file_abs_path.parent.as_posix())

from config import model_config


class Environment(object):
    def __init__(self):
        for key, value in model_config.items():
            self.__setattr__(key, value)


class ExpermentLab(Environment):
    """
    Class to control whole model training pipeline
    """

    def __init__(self, *args, **kwargs):
        super(ExpermentLab, self).__init__(*args, **kwargs)


class Data(Environment):
    """
    Class to handle all data related process
    like
        - read raw data and convert it to HDF5 file system
        - Convert HDF5 file to TF records
        - Convert TF records to TF dataset objects
        - Convert TF dataset objects to (features, labels) array tuples
    """

    def __init__(self, *args, **kwargs):
        super(Data, self).__init__(*args, **kwargs)

    def read_image_data(self) -> tuple:
        """
        Function to convert raw train and test images data to HDF5 data
        returns path to train and test hdf5 data files
        return : tuple(Path, Path)
        """
        # list all class images from train and test
        train_images = self.train_data_dir.glob('*/*')
        test_images = self.test_data_dir.glob('*/*')

        # convert these images to hdf5
        train_data = Images2HDF5File(images_list_gen=train_images,
                                     hdf5_file_save_path=self.hdf5_train_data
                                     )
        test_data = Images2HDF5File(images_list_gen=test_images,
                                    hdf5_file_save_path=self.hdf5_test_data
                                    )
        return train_data, test_data

    def read_raw_data(self) -> tuple:
        """
        Method to read raw data either numpy or images
        and convert it to HDF5 file.
        return: [(np.array, Path), (np.array, Path)]
        """

        # if data is in numpy format
        if self.dataFormat=="numpy":
            # convert numpy to hdf5 data
            (x_train, y_train), (x_test, y_test) = self.dataSource
            train_dataset = numpy2hdf5(x_train, y_train, self.hdf5_train_data)
            test_dataset =  numpy2hdf5(x_test, y_test, self.hdf5_test_data)

        # if data is in raw image format
        if self.dataFormat=="image":
            train_dataset, test_dataset = self.read_image_data()
        return train_dataset, test_dataset


    def augment_data(self, data: Path) -> Path:
        """
        Method to create augmented data for  hdf5 file
        data: Path object for train hdf5 file
        It creates augmentation based on on the data_augmentation_fraction
        provided in config, ie it augments x% of images from actual images
        """
        augmented_data_hdf5 = data
        if not augmented_data_hdf5.exists():
            # create hdf5 file to save augmented data
            augmented_data = tables.open_file(augmented_data_hdf5.as_posix(), mode='w')

            augmented_images_earry = augmented_data.create_earray(where=augmented_data.root,
                                                                  name='images',
                                                                  atom=tables.UInt8Atom(),
                                                                  shape=[0,
                                                                         model_config['img_height'],
                                                                         model_config['img_width'],
                                                                         model_config['img_depth']]
                                                                  )
            augmented_lables_earry = augmented_data.create_earray(where=augmented_data.root,
                                                                  name='labels',
                                                                  atom=tables.UInt8Atom(),
                                                                  shape=[0,1]
                                                                  )

            # check whether data is numpy or Path object
            hdf5File = tables.open_file(self.hdf5_train_data, mode='r')
            images = hdf5File.root.images
            labels = hdf5File.root.labels


            # apply augmentations on data
            self.total_aug_images = 0
            for augmentation_func in self.data_augmentation_functions:
                aug_imgs, aug_labels = map_augmentation(augmentation_func,
                                                        images,
                                                        labels,
                                                        fraction=self.data_augmentation_fraction)
                self.total_aug_images += len(aug_imgs)

                # dump augmented data to train_data_aug_hdf5 file
                for aug_image, aug_label in zip(aug_imgs, aug_labels):
                    augmented_images_earry.append(aug_image[None])
                    augmented_lables_earry.append(aug_label[None])

                # add original data also to train_data_aug_hdf5
                for image, label in zip(images, labels):
                    augmented_images_earry.append(image[None])
                    augmented_lables_earry.append(label[None])

            print(f'Total number of augmentations on train data:{self.total_aug_images}')

            # close hdf5 file
            augmented_data.close()
            return augmented_data
        else:
            print(f'{augmented_data_hdf5.as_posix()} already exists')

    def createTfrecords(self)->tuple:
        """
        Method to convert train and test hdf5 files to Tfrecords
        :return: (train_records_path, test_records_path)
        """
        train_records = HDF52Tfrecords(self.hdf5_train_data, Tfrecords_save_path=self.train_tfrecords)
        test_records = HDF52Tfrecords(self.hdf5_test_data, Tfrecords_save_path=self.test_tfrecords)
        return train_records, test_records

    def createTfdatasets(self)->tuple:
        trainDataset = Tfrecords2TfDatasets(record=self.train_tfrecords)
        testDataset = Tfrecords2TfDatasets(record=self.test_tfrecords)
        return trainDataset, testDataset

if __name__ == "__main__":
    o = Data()
    # o.augment_data(data=Path('/Users/satishjasthi/Documents/Professional/ML/drshti_yantrikarana/HDF5Data/cifar10_test.h5'))
    o.createTfdatasets()
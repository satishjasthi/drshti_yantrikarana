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

from Utils.dataUtils import Images2HDF5File

current_file_abs_path = Path(__file__).resolve()
sys.path.append(current_file_abs_path.parent.as_posix())

from config import model_config

class ExpermentLab(object):
    """
    Class to control whole model training pipeline
    """

    def __init__(self):
        for key, value in model_config.items():
            self.__setattr__(key, value)

    def read_raw_data(self)-> tuple[np.array, np.array]:
        """
        Method to read raw data and convert it to HDF5 file
        """
        # if data is in numpy format
        if isinstance(self.dataFormat, np.ndarray):
            # (x_train, y_train), (x_test, y_test) =
            pass

        # if data is in raw image format
        if isinstance(self.dataFormat, Path):
            pass

class Data(ExpermentLab):
    """
    Class to handle all data related process
    like
        - read raw data and convert it to HDF5 file system
        - Convert HDF5 file to TF records
        - Convert TF records to TF dataset objects
        - Convert TF dataset objects to (features, labels) array tuples
    """

    def __init__(self, *args, **kwargs):
        super(Data,  self).__init__(*args, **kwargs)




    def read_image_data(self)->tuple[(tables.earray, tables.earray),
                                     (tables.earray, tables.earray)]:
        """
        Function to convert raw train and test images data to HDF5 data
        """
        # list all class images from train and test
        train_images = self.train_data_dir.glob('*/*')
        test_images = self.test_data_dir.glob('*/*')

        # convert these images to hdf5
        x_train, y_train = Images2HDF5File(images_list_gen=train_images,
                                           hdf5_file_save_path=self.hdf5_train_data
                                           )
        x_test, y_test = Images2HDF5File(images_list_gen=test_images,
                                         hdf5_file_save_path=self.hdf5_test_data
                                         )
        return ((x_train, y_train), (x_test, y_test))


if __name__ == "__main__":
    o = ExpermentLab()
    print(dir(o))
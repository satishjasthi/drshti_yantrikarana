"""
Script to includes utility functions to do data processing

author: Satish Jasthi
------
"""
from pathlib import Path

import numpy as np
import tables
from PIL import Image
from config import model_config


def data_check(new_x: np.array,
               new_y: np.array,
               old_x: np.array,
               old_y: np.array,
               ) -> bool:
    """
    Function to check if the size of numpy array data before and after some
    transformation/process  are same
    :param new_x:
    :param new_y:
    :param old_x:
    :param old_y:
    :return:
    """

    if new_x.shape != old_x.shape:
        return False
    if new_y.shape != old_y.shape:
        return False
    return True


def Images2HDF5File(images_list_gen: Path.glob = None,
                    hdf5_file_save_path: Path = None) -> tuple[tables.earray, tables.earray]:
    """
    Function to convert list of images in train or test dir into a single
    numpy array based HDF5 file

    data in train or test is arranged in following way

    train
        class1
            image1
        class2
            image2

    :param images_list_gen: is generator that yields pathlib path object to image
    :return:
    """
    hdf5_file = tables.open_file(hdf5_file_save_path.as_posix(), mode='w')
    images_earray = hdf5_file.create_earray(where=hdf5_file.root,
                                            name='images',
                                            atom=tables.UInt8Atom(),
                                            shape=[0,
                                                   model_config['img_height'],
                                                   model_config['img_width'],
                                                   model_config['img_depth']]
                                            )

    classes = list()
    for image in images_list_gen:
        class_name = image.parent
        pil_image = Image.open(image)
        img_arr = np.array(pil_image)
        img_arr = img_arr[:, :, :3]
        images_earray.append(img_arr[None])
        classes.append(class_name)

    unique_classes = sorted(list(set(classes)))
    class_index_map = {class_name: index for index, class_name in enumerate(unique_classes)}
    labels = [class_index_map[cls] for cls in classes]

    assert len(labels) == len(classes), "Len of labels not matching number of samples in data!!!"

    labels_array = hdf5_file.create_array(where=hdf5_file.root,
                                          name='labels',
                                          obj=labels
                                          )
    return images_earray, labels_array

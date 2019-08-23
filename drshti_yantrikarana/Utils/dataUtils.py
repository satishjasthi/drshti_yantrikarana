"""
Script to includes utility functions to do data processing

author: Satish Jasthi
------
"""
import random
from collections import Callable
from multiprocessing import cpu_count
from pathlib import Path

import numpy as np
import tables
from PIL import Image
import tensorflow as tf
from tensorflow.python import keras

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


def numpy2hdf5(images: np.array = None,
               labels: np.array = None,
               hdf5_file_save_path: Path = None) -> Path:
    """
    Function to convert image and label numpy arrays to hdf5 file
    """
    if not hdf5_file_save_path.exists():
        # open hdf5 file
        hdf5_file = tables.open_file(hdf5_file_save_path.as_posix(), mode='w')

        # create earrays for images and labels
        images_earray = hdf5_file.create_earray(where=hdf5_file.root,
                                                name='images',
                                                atom=tables.UInt8Atom(),
                                                shape=[0,
                                                       model_config['img_height'],
                                                       model_config['img_width'],
                                                       model_config['img_depth']]
                                                )
        labels_earray = hdf5_file.create_earray(where=hdf5_file.root,
                                                name='labels',
                                                atom=tables.UInt8Atom(),
                                                shape=[0, 1]
                                                )

        # dump numpy arrays to hdf5 file
        for image, label in zip(images, labels):
            images_earray.append(image[None])
            labels_earray.append(label[None])

        # close hdf5 file
        hdf5_file.close()
        return hdf5_file_save_path
    else:
        print(f'{hdf5_file_save_path.as_posix()} already exists')
        return hdf5_file_save_path


def Images2HDF5File(images_list_gen: Path.glob = None,
                    hdf5_file_save_path: Path = None) -> Path:
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
    if not hdf5_file_save_path.exists():
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
            print(image)
            class_name = image.parent
            pil_image = Image.open(image)
            img_arr = np.array(pil_image)
            print(img_arr.shape)
            assert len(img_arr.shape) < 4, "Image has more than 3 dimensions"

            img_arr = img_arr[:, :, :3]
            images_earray.append(img_arr[None])
            classes.append(class_name)

        unique_classes = sorted(list(set(classes)))
        class_index_map = {class_name: index for index, class_name in enumerate(unique_classes)}
        labels = [class_index_map[cls] for cls in classes]

        assert len(labels) == len(classes), "Len of labels not matching number of samples in data!!!"

        # add labels to hdf5 data
        hdf5_file.create_array(where=hdf5_file.root,
                               name='labels',
                               obj=labels
                               )
        hdf5_file.close()
    else:
        print(f'{hdf5_file_save_path.as_posix()} already exists, Returning file Path object')
    return hdf5_file_save_path


def map_augmentation(func: Callable,
                     img_array: np.array,
                     label_array: np.array,
                     fraction: float) -> (np.array, np.array):
    """
    Function to map func on a portion of np array
    :return:
    """
    assert len(img_array.shape) == 4, f"map augmentation function needs img_array to be in 4D, " \
        f"but got {len(img_array.shape)}"

    # calculate number of images to be augmented
    num_aug_images = int(round(fraction * img_array.shape[0]))

    # shuffle and select images for augmentation
    img_indexes = list(range(0, img_array.shape[0]))
    random.shuffle(img_indexes)

    aug_images, aug_labels = list(), list()
    for i in range(num_aug_images):
        index = random.choice(img_indexes)

        # remove index to avoid data repetation
        img_indexes.remove(index)

        image, label = img_array[index, :, :, :], label_array[index]
        aug_images.append(func(image))
        aug_labels.append(label)

    assert len(aug_images) == num_aug_images, "Number of aug images is not equal to requested aug images"
    assert len(aug_labels) == num_aug_images, "Number of aug images is not equal to requested aug images"

    return aug_images, aug_labels


def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def convert2FeatureMessage(image_array: np.array, label: int):
    """
    Method to convert image array to serialized tf.train.Feature message
    :param image_array: np.array
    :param label: int
    :return:
    """
    image_string = tf.io.serialize_tensor(tf.convert_to_tensor(image_array, dtype=tf.uint8))
    image_shape = image_array.shape

    feature = {
        'height': _int64_feature(image_shape[0]),
        'width': _int64_feature(image_shape[1]),
        'depth': _int64_feature(image_shape[2]),
        'label': _int64_feature(label),
        'image_raw': _bytes_feature(image_string.numpy()),
    }
    return tf.train.Example(features=tf.train.Features(feature=feature))

def HDF52Tfrecords(data_dir:Path, Tfrecords_save_path:Path)->Path:
    """
    Function to convert image and label numpy data in hdf5 file to TF records
    :param data_dir: hdf5 file path which has images and labels
    :return: tf records path object
    """
    # read hdf5 data file
    hdf5data = tables.open_file(data_dir, mode='r')
    images = hdf5data.root.images
    np_labels = hdf5data.root.labels

    print(f'Number of images in {data_dir.name}: {images.shape}')
    print(f'Number of labels in {data_dir.name}: {np_labels.shape}')

    assert isinstance(np_labels[0], np.ndarray), 'Labels are not in [label] format'
    num_classes = model_config['numClasses']

    # write tf record file
    with tf.compat.v1.python_io.TFRecordWriter(Tfrecords_save_path.as_posix()) as writer:
        for image_arr, label in zip(images, np_labels):
            tf_example = convert2FeatureMessage(image_arr, label)
            writer.write(tf_example.SerializeToString())
    return Tfrecords_save_path


def _parse_image_function(example_proto, one_hot_encoding=True):
    # Create a dictionary describing the features.
    image_feature_description = {
        'height': tf.io.FixedLenFeature([], tf.int64),
        'width': tf.io.FixedLenFeature([], tf.int64),
        'depth': tf.io.FixedLenFeature([], tf.int64),
        'label': tf.io.FixedLenFeature([], tf.int64),
        'image_raw': tf.io.FixedLenFeature([], tf.string),
    }

    # Parse the input tf.Example proto using the dictionary above.
    parsed_dataset = tf.io.parse_single_example(example_proto, image_feature_description)

    # read in raw format
    image = tf.io.parse_tensor(parsed_dataset['image_raw'], out_type=tf.uint8)
    # label = tf.io.parse_tensor(parsed_dataset['label'], out_type=tf.int64)

    # change both image and label dtypes to float to use preprocess functions
    # image = tf.cast(image, dtype=tf.float64)
    # label = tf.cast(label, dtype=tf.float64)

    image_shape = [parsed_dataset['height'], parsed_dataset['width'], parsed_dataset['depth']]
    image = tf.reshape(image, image_shape)

    if one_hot_encoding:
        one_hot_encd_label = tf.one_hot(parsed_dataset['label'], depth=model_config['numClasses'])
        return image, one_hot_encd_label
    else:
        return image, parsed_dataset['label']


def readTfRecord(tfrecords_file:Path):
    """
    Method to read tf records which has image and labels
    :return:parsed_dataset: is a tuple(image, label)
    """

    raw_dataset = tf.data.TFRecordDataset(tfrecords_file.as_posix())
    parsed_dataset = raw_dataset.map(_parse_image_function)
    return parsed_dataset

def preprocessTfDataset(image, label):

    # normalize
    image = image/ 255.0

    # standardize
    mu_tensor, std_tensor = tf.convert_to_tensor([0.4914, 0.4822, 0.4465]), tf.convert_to_tensor([0.2470, 0.2435, 0.2616])
    image = (image - tf.cast(mu_tensor, dtype=tf.float64))/ tf.cast(std_tensor,dtype=tf.float64)
    return image, label

def Tfrecords2TfDatasets(record:Path)->tf.data.Dataset:
    """
    Convert tfrecords to tf dataset objects
    :param records: path object to tf records file
    """
    # read tf record
    parsed_data = readTfRecord(tfrecords_file=record)

    # shuffle records
    if model_config['tfDatasetShuffleBool']:
        dataset = parsed_data.shuffle(buffer_size=model_config['batchSize'])

    # dataset = dataset.map(preprocessTfDataset)
    dataset = dataset.batch(batch_size=model_config['batchSize'])
    if model_config['tfDatasetPrefetchBool']:
        dataset = dataset.prefetch(buffer_size=model_config['prefetchBufferSize'])
    return dataset
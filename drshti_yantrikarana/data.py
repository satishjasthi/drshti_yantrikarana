"""
Reference:
    - Augmentation methods are taken from fenwickslab
Usage:

About:

Author: Satish Jasthi
"""
from pathlib import Path

import tables
import numpy as np
import pandas as pd
import tensorflow as tf

# Data ingestion........................................................................................................
from PIL import Image

trainTestSplitDir = ''

class TfRecords(object):
    """
    Class to create TfRecords
    mode: train ot

    """

    def __init__(self, mode:str=None,
                 TrainHdf5_data:Path=None,
                 TestHdf5_data:Path=None,
                 TrainTfRecord_data=None,
                 TestTfRecord_data=None
                 ):
        self.mode = mode
        self.TrainHdf5_data = TrainHdf5_data
        self.TestHdf5_data  =TestHdf5_data
        self.TrainTfRecord_data = TrainTfRecord_data
        self.TestTfRecord_data = TestTfRecord_data

        if mode == 'train':
            if self.TrainHdf5_data.exists():
                self.hdf5_data = tables.open_file(self.TrainHdf5_data.as_posix(), mode='r')
            else:
                raise IOError(
                    f'Unable to find {TrainHdf5_data.as_posix()}, Please create hdf5 file before running tf records')
        elif mode == 'test':
            if self.TestHdf5_data.exists():
                self.hdf5_data = tables.open_file(self.TestHdf5_data.as_posix(), mode='r')
            else:
                raise IOError(
                    f'Unable to find {TestHdf5_data.as_posix()}, Please create hdf5 file before running tf records')

    @staticmethod
    def _int64_feature(value):
        """Returns an int64_list from a bool / enum / int / uint."""
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

    @staticmethod
    def _bytes_feature(value):
        """Returns a bytes_list from a string / byte."""
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    def convert2FeatureMessage(self, image_array: np.array, label: int):
        """
        Method to convert image array to serialized tf.train.Feature message
        :param image_array: np.array
        :param label: int
        :return:
        """
        image_string = tf.io.serialize_tensor(tf.convert_to_tensor(image_array, dtype=tf.uint8))
        image_shape = image_array.shape

        feature = {
            'height': self._int64_feature(image_shape[0]),
            'width': self._int64_feature(image_shape[1]),
            'depth': self._int64_feature(image_shape[2]),
            'label': self._int64_feature(label),
            'image_raw': self._bytes_feature(image_string.numpy()),
        }
        return tf.train.Example(features=tf.train.Features(feature=feature))

    def writeTfRecord(self):
        """
        Method to create TFRecord for data set
        :return:
        """
        # read images and labels from hdf5
        if self.mode == 'train':
            hdf5_data_arrays = tables.open_file(self.TrainHdf5_data.as_posix(), w='r')
        elif self.mode == 'test':
            hdf5_data_arrays = tables.open_file(self.TestHdf5_data.as_posix(), w='r')
        else:
            raise NotImplementedError(f'mode: {self.mode} is not valid, please choose mode between (train, test)')

        # image generator
        image_gen = hdf5_data_arrays.root.images

        # create label index map
        label_names = hdf5_data_arrays.root.labels[:]
        unique_labels = set(label_names)
        label_index_map = {label: index for index, label in enumerate(unique_labels)}

        if self.mode == 'train':
            with tf.compat.v1.python_io.TFRecordWriter(self.TrainTfRecord_data.as_posix()) as writer:
                for image_arr, label in zip(image_gen, label_names):
                    tf_example = self.convert2FeatureMessage(image_arr, label_index_map[label])
                    writer.write(tf_example.SerializeToString())

            hdf5_data_arrays.close()
        elif self.mode == 'test':
            with tf.compat.v1.python_io.TFRecordWriter(self.TestTfRecord_data.as_posix()) as writer:
                for image_arr, label in zip(image_gen, label_names):
                    tf_example = self.convert2FeatureMessage(image_arr, label_index_map[label])
                    writer.write(tf_example.SerializeToString())

            hdf5_data_arrays.close()

    @staticmethod
    def _parse_image_function(example_proto):
        # Create a dictionary describing the features.
        image_feature_description = {
            'height': tf.io.FixedLenFeature([], tf.int64),
            'width': tf.io.FixedLenFeature([], tf.int64),
            'depth': tf.io.FixedLenFeature([], tf.int64),
            'label': tf.io.FixedLenFeature([], tf.int64),
            'image_raw': tf.io.FixedLenFeature([], tf.string),
        }

        # Parse the input tf.Example proto using the dictionary above.
        return tf.io.parse_single_example(example_proto, image_feature_description)

    def readTfRecord(self):
        """
        Method to read tf record format image data
        :return:
        """
        if self.mode == 'train':
            raw_image_dataset = tf.data.TFRecordDataset(self.TrainTfRecord_data)
        elif self.mode == 'test':
            raw_image_dataset = tf.data.TFRecordDataset(self.TestTfRecord_data)
        else:
            raise NotImplementedError(f'mode: {self.mode} is not valid, please choose mode between (train, test)')
        parsed_image_dataset = raw_image_dataset.map(self._parse_image_function)
        return parsed_image_dataset


class ConvertImages2Hdf5():
    """
    Class to convert images to hdf5 file system by saving images and labels as ndarrays
    as shown below:
        root
            images:
            labels:

    data_dir: has data stored class wise as shown below
        train
            class1
                img1

        test
            class1
                imgt1
    """

    def __init__(self, data_dir:Path, hdf5_save_dir:Path, resizeHeight:int, resizeWidth:int):
        self.data_dir = data_dir
        self.train_dir = self.data_dir/'train'
        self.test_dir = self.data_dir / 'test'
        self.hdf5_save_dir = hdf5_save_dir
        self.resizeHeight = resizeHeight
        self.resizeWidth = resizeWidth


    def createHdf5File(self, directory):
        """

        :param directory: Path, can be train or test dir
        :return: None
        """
        hdf5_save_file = self.hdf5_save_dir/f'{directory.name}.h5'
        hdf5_save_file_obj = tables.open_file(hdf5_save_file.as_posix(), mode='w')

        # create expandable arrays to store images and their labels as arrays
        images_earray = hdf5_save_file_obj.create_earray(where=hdf5_save_file_obj.root,
                                                     name='images',
                                                     atom=tables.UInt8Atom(),
                                                     shape=[0, self.resizeHeight, self.resizeWidth, 3])
        labels_erray = hdf5_save_file_obj.create_earray(where=hdf5_save_file_obj.root,
                                                     name='labels',
                                                     atom=tables.UInt8Atom(),
                                                     shape=[0, 1])

        classes = sorted(list(directory.glob('*')))

        # save class index map to hdf5 file
        class_index_df = pd.DataFrame(data={'class':classes,
                           'label_index':list(range(0, len(classes)))
                           }
                     )
        class_index_df.to_hdf(path_or_buf=hdf5_save_file.as_posix(), key='/class_index_df')

        for image in directory.glob('*/*'):
            class_name = image.parent
            class_index = class_index_df[class_index_df['class']==class_name]['label_index']

            # read image and resize it
            img = np.array(Image.open(image))
            rs_img = resize_image(img_tensor=tf.convert_to_tensor(img),
                                  resize_shape=(self.resizeHeight, self.resizeWidth)
                                  ).numpy()

            # add image and label to hdf5 earray
            images_earray.append(rs_img)
            labels_erray.append(class_index)
        hdf5_save_file_obj.close()

    def createTrainTestHdf5Files(self):
        self.createHdf5File(directory=self.train_dir)
        self.createHdf5File(directory=self.test_dir)


# preprocessing methods
def resize_image(img_tensor: tf.convert_to_tensor, resize_shape:tuple) -> tf.image:
    """
    Function to read an image, convert it into a square image and
    resize it to standard resolution as defined in config
    :param image_path:tf.convert_to_tensor
    :param resize_shape: tuple (h,w)
    :return: tf.image
    """
    # create a central crop wrt larger side to create square image
    h, w = tf.shape(img_tensor)[:2]
    if h > w:
        cropped_image = tf.image.crop_to_bounding_box(img_tensor,
                                                      offset_height=(h - w) // 2,
                                                      offset_width=0,
                                                      target_height=w,
                                                      target_width=w
                                                      )
    else:
        cropped_image = tf.image.crop_to_bounding_box(img_tensor,
                                                      offset_height=0,
                                                      offset_width=(w - h) // 2,
                                                      target_height=h,
                                                      target_width=h
                                                      )

    # resize image to predifined res in config
    resized_image = tf.image.resize(cropped_image, size=resize_shape)
    return resized_image



# define data augmentations

def replace_slice(input_: tf.Tensor, replacement, begin) -> tf.Tensor:
    inp_shape = tf.shape(input_)
    size = tf.shape(replacement)
    padding = tf.stack([begin, inp_shape - (begin + size)], axis=1)
    replacement_pad = tf.pad(replacement, padding)
    mask = tf.pad(tf.ones_like(replacement, dtype=tf.bool), padding)
    return tf.where(mask, replacement_pad, input_)


def cutout(x: tf.Tensor, h: int, w: int, c: int = 3) -> tf.Tensor:
    """
    Cutout data augmentation. Randomly cuts a h by w whole in the image, and fill the whole with zeros.
    :param x: Input image.
    :param h: Height of the hole.
    :param w: Width of the hole
    :param c: Number of color channels in the image. Default: 3 (RGB).
    :return: Transformed image.
    """
    shape = tf.shape(x)
    x0 = tf.random.uniform([], 0, shape[0] + 1 - h, dtype=tf.int32)
    y0 = tf.random.uniform([], 0, shape[1] + 1 - w, dtype=tf.int32)
    x = replace_slice(x, tf.zeros([h, w, c]), [x0, y0, 0])
    return x

def random_flip(x: tf.Tensor, flip_vert: bool = False) -> tf.Tensor:
    """
    Randomly flip the input image horizontally, and optionally also vertically, which is implemented as 90-degree
    rotations.
    :param x: Input image.
    :param flip_vert: Whether to perform vertical flipping. Default: False.
    :return: Transformed image.
    """
    x = tf.image.random_flip_left_right(x)
    if flip_vert:
        x = random_rotate_90(x)
    return x


def random_rotate_90(x: tf.Tensor) -> tf.Tensor:
    """
    Randomly rotate the input image by either 0, 90, 180 or 270 degrees.
    :param x: Input image.
    :return: Transformed image.
    """
    return tf.image.rot90(x, tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32))

def deg2rad(x: tf.Tensor) -> tf.Tensor:
    """
    Converts an angle in degrees to radians.
    :param x: Input angle, in degrees.
    :return: Angle in radians
    """
    return (x * np.pi) / 180

def random_rotate_matrix(max_deg: float = 10) -> tf.Tensor:
    deg = tf.random.uniform(shape=[], minval=-max_deg, maxval=max_deg, dtype=tf.float32)
    rad = deg2rad(deg)
    return tf.convert_to_tensor([[tf.cos(rad), -tf.sin(rad), 0], [tf.sin(rad), tf.cos(rad), 0], [0, 0, 1]])


def affine_grid_generator(H: int, W: int, tfm_mat) -> tf.Tensor:
    B = tf.shape(tfm_mat)[0]

    x = tf.linspace(-1.0, 1.0, W)
    y = tf.linspace(-1.0, 1.0, H)
    x_t, y_t = tf.meshgrid(x, y)

    x_t_flat = tf.reshape(x_t, [-1])
    y_t_flat = tf.reshape(y_t, [-1])

    # reshape to [x_t, y_t , 1] - (homogeneous form)
    ones = tf.ones_like(x_t_flat)
    sampling_grid = tf.stack([x_t_flat, y_t_flat, ones])

    # repeat grid B times
    sampling_grid = tf.expand_dims(sampling_grid, axis=0)
    sampling_grid = tf.tile(sampling_grid, tf.stack([B, 1, 1]))

    # cast to float32 (required for matmul)
    tfm_mat = tf.cast(tfm_mat, tf.float32)
    sampling_grid = tf.cast(sampling_grid, tf.float32)

    # transform the sampling grid - batch multiply
    batch_grids = tf.matmul(tfm_mat, sampling_grid)
    # batch grid has shape (B, 2, H*W)

    # reshape to (B, H, W, 2)
    batch_grids = tf.reshape(batch_grids, [B, 2, H, W])

    return batch_grids

def reflect(x, max_x):
    x = tf.abs(x)
    x = max_x - tf.abs(max_x - x)
    return x

def get_pixel_value(img: tf.Tensor, x, y) -> tf.Tensor:
    x_shape = tf.shape(x)
    B = x_shape[0]
    H = x_shape[1]
    W = x_shape[2]

    batch_idx = tf.range(0, B)
    batch_idx = tf.reshape(batch_idx, (B, 1, 1))
    b = tf.tile(batch_idx, (1, H, W))

    indices = tf.stack([b, y, x], 3)
    return tf.gather_nd(img, indices)

def bilinear_sampler(img: tf.Tensor, x, y, do_reflect: bool = False) -> tf.Tensor:
    img_shape = tf.shape(img)
    H = img_shape[1]
    W = img_shape[2]

    max_y = tf.cast(H - 1, tf.int32)
    max_x = tf.cast(W - 1, tf.int32)
    zero = tf.zeros([], dtype=tf.int32)

    # rescale x and y to [0, W-1/H-1]
    x = tf.cast(x, tf.float32)
    y = tf.cast(y, tf.float32)
    x = 0.5 * ((x + 1.0) * tf.cast(max_x - 1, tf.float32))
    y = 0.5 * ((y + 1.0) * tf.cast(max_y - 1, tf.float32))

    # grab 4 nearest corner points for each (x_i, y_i)
    x0 = tf.cast(tf.floor(x), tf.int32)
    x1 = x0 + 1
    y0 = tf.cast(tf.floor(y), tf.int32)
    y1 = y0 + 1

    if do_reflect:
        x0 = reflect(x0, max_x)
        x1 = reflect(x1, max_x)
        y0 = reflect(y0, max_y)
        y1 = reflect(y1, max_y)

    # clip to range [0, H-1/W-1] to not violate img boundaries
    x0 = tf.clip_by_value(x0, zero, max_x)
    x1 = tf.clip_by_value(x1, zero, max_x)
    y0 = tf.clip_by_value(y0, zero, max_y)
    y1 = tf.clip_by_value(y1, zero, max_y)

    # get pixel value at corner coords
    Ia = get_pixel_value(img, x0, y0)
    Ib = get_pixel_value(img, x0, y1)
    Ic = get_pixel_value(img, x1, y0)
    Id = get_pixel_value(img, x1, y1)

    # recast as float for delta calculation
    x0 = tf.cast(x0, tf.float32)
    x1 = tf.cast(x1, tf.float32)
    y0 = tf.cast(y0, tf.float32)
    y1 = tf.cast(y1, tf.float32)

    # calculate deltas
    wa = (x1 - x) * (y1 - y)
    wb = (x1 - x) * (y - y0)
    wc = (x - x0) * (y1 - y)
    wd = (x - x0) * (y - y0)

    # add dimension for addition
    wa = tf.expand_dims(wa, axis=3)
    wb = tf.expand_dims(wb, axis=3)
    wc = tf.expand_dims(wc, axis=3)
    wd = tf.expand_dims(wd, axis=3)

    # compute output
    out = tf.add_n([wa * Ia, wb * Ib, wc * Ic, wd * Id])

    return out

def affine_transform(X: tf.Tensor, tfm_mat, out_dims=None, do_reflect: bool = False) -> tf.Tensor:
    X_shape = tf.shape(X)
    B = X_shape[0]
    H = X_shape[1]
    W = X_shape[2]
    tfm_mat = tf.reshape(tfm_mat, [B, 2, 3])

    out_H, out_W = out_dims if out_dims else H, W
    batch_grids = affine_grid_generator(out_H, out_W, tfm_mat)

    x_s = batch_grids[:, 0, :, :]
    y_s = batch_grids[:, 1, :, :]

    return bilinear_sampler(X, x_s, y_s, do_reflect)

def apply_affine_mat(x: tf.Tensor, mat: tf.Tensor, do_reflect: bool = False) -> tf.Tensor:
    mat = tf.reshape(mat, [-1])[:6]
    x = tf.expand_dims(x, 0)
    x = affine_transform(x, mat, do_reflect)
    x = tf.clip_by_value(x, 0.0, 1.0)
    x = tf.squeeze(x, [0])
    return x

def random_rotate(x: tf.Tensor, max_rot_deg: float = 10) -> tf.Tensor:
    mat = random_rotate_matrix(max_rot_deg)
    return apply_affine_mat(x, mat)




# map data augmentations

# save augmented data as TF records

"""
Reference:
    - Augmentation methods are taken from fenwickslab
Usage:

About:

Author: Satish Jasthi
"""
import logging
from pathlib import Path

import tables
import numpy as np
import pandas as pd
import tensorflow as tf

# Data ingestion........................................................................................................
from PIL import Image
from tensorflow.python import keras

from moduleLogger import DyLogger
import time
logger = DyLogger(logging_level=logging.DEBUG)
# logger.get_logger(__name__)

def timeme(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        if 'log_time' in kw:
            name = kw.get('log_name', method.__name__.upper())
            kw['log_time'][name] = int((te - ts) * 1000)
        else:
            print ('%r  %2.2f ms' % \
                  (method.__name__, (te - ts) * 1000))
        return result
    return timed

class TFRecords(object):
    """
    Class to create TFRecords
    mode: train ot

    """

    def __init__(self, mode:str=None,
                 TrainHdf5_data:Path=None,
                 TestHdf5_data:Path=None,
                 TrainTfRecord_data=None,
                 TestTfRecord_data=None,
                 num_classes=None
                 ):
        self.mode = mode
        self.num_classes=num_classes
        self.logger = logger.get_logger(__name__)
        self.TrainHdf5_data = TrainHdf5_data
        self.TestHdf5_data  =TestHdf5_data
        self.TrainTfRecord_data = TrainTfRecord_data
        self.TestTfRecord_data = TestTfRecord_data

        if self.mode == 'train':
            self.logger.debug(f'Reading TrainHdf5_data from :{self.TrainHdf5_data.as_posix()}')
            self.logger.debug(f'Saving TrainTfRecord_data from :{self.TrainTfRecord_data.as_posix()}')
        if self.mode == 'test':
            self.logger.debug(f'Reading TestHdf5_data from :{self.TestHdf5_data.as_posix()}')
            self.logger.debug(f'Saving TestTfRecord_data from :{self.TestTfRecord_data.as_posix()}')


        self.logger.debug(f'Creating TF records for {self.mode} data')
        if self.mode == 'train':
            if self.TrainHdf5_data.exists():

                self.hdf5_data = tables.open_file(self.TrainHdf5_data.as_posix(), mode='r')
            else:
                self.logger.error('Error occured: '+ f'Unable to find {TrainHdf5_data.as_posix()}, '
                                                     f'Please create {TrainHdf5_data.parent.name} Folder before running'
                                                     f' tf records')
                raise IOError(
                    f'Unable to find {TrainHdf5_data.as_posix()}, Please create {TrainHdf5_data.parent.name} Folder before running tf records')
        elif self.mode == 'test':
            if self.TestHdf5_data.exists():
                self.hdf5_data = tables.open_file(self.TestHdf5_data.as_posix(), mode='r')
            else:
                self.logger.error('Error occured: ' + f'Unable to find {TrainHdf5_data.as_posix()}, '
                                                      f'Please create {TrainHdf5_data.parent.name} Folder '
                                                      f'before running tf records')
                raise IOError(
                    f'Unable to find {TestHdf5_data.as_posix()}, Please create {TrainHdf5_data} Folder before running tf records')

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

    @timeme
    def writeTfRecord(self):
        """
        Method to create TFRecord for data set
        :return:
        """
        # read images and labels from hdf5
        self.logger.debug(f'Creating {self.mode} train records')
        if self.mode == 'train':
            hdf5_data_arrays = tables.open_file(self.TrainHdf5_data.as_posix(), w='r')
        elif self.mode == 'test':
            hdf5_data_arrays = tables.open_file(self.TestHdf5_data.as_posix(), w='r')
        else:
            self.logger.error('Error occured: '+ f'mode: {self.mode} is not valid, please choose mode between '
            f'(train, test)')
            raise NotImplementedError(f'mode: {self.mode} is not valid, please choose mode between (train, test)')

        # image generator
        image_gen = hdf5_data_arrays.root.images

        # create label index map
        label_names = np.squeeze(hdf5_data_arrays.root.labels)
        # unique_labels = list(np.unique(label_names))
        unique_labels = list(range(0, self.num_classes))
        label_index_map = {label: index for index, label in enumerate(unique_labels)}

        self.logger.debug(f'Saving {self.mode} tf record files')
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

    # @staticmethod
    @timeme
    def _parse_image_function(self,example_proto):
        # Create a dictionary describing the features.
        image_feature_description = {
            'height': tf.io.FixedLenFeature([], tf.int64),
            'width': tf.io.FixedLenFeature([], tf.int64),
            'depth': tf.io.FixedLenFeature([], tf.int64),
            'label': tf.io.FixedLenFeature([], tf.int64),
            'image_raw': tf.io.FixedLenFeature([], tf.string),
        }

        # Parse the input tf.Example proto using the dictionary above.
        parsed_dataset =  tf.io.parse_single_example(example_proto, image_feature_description)
        image = tf.io.parse_tensor(parsed_dataset['image_raw'], out_type=tf.uint8)
        image_shape = [parsed_dataset['height'], parsed_dataset['width'], parsed_dataset['depth']]
        image = tf.reshape(image, image_shape)
        label_one_hot = tf.one_hot(parsed_dataset['label'], depth=self.num_classes)
        return image, label_one_hot

    @timeme
    def readTfRecord(self):
        """
        Method to read tf record format image data
        :return:
        """
        self.logger.debug(f'Reading {self.mode} tf record files')
        if self.mode == 'train':
            raw_image_dataset = tf.data.TFRecordDataset(self.TrainTfRecord_data.as_posix())
        elif self.mode == 'test':
            raw_image_dataset = tf.data.TFRecordDataset(self.TestTfRecord_data.as_posix())
        else:
            self.logger.error('Error occured' + f'mode: {self.mode} is not valid, please choose mode between '
            f'(train, test)')
            raise NotImplementedError(f'mode: {self.mode} is not valid, please choose mode between (train, test)')
        parsed_dataset = raw_image_dataset.map(self._parse_image_function)

        return parsed_dataset

@timeme
def preprocessTfDataset(image_label_instance):
    image = tf.image.decode_image(image_label_instance['image_raw'], dtype=tf.float64)
    label = image_label_instance['label']

    # normalize
    image = image/ 255.0

    # standardize
    mu_tensor, std_tensor = tf.convert_to_tensor([0.4914, 0.4822, 0.4465]), tf.convert_to_tensor([0.2470, 0.2435, 0.2616])
    image = (image - tf.cast(mu_tensor, dtype=tf.float64))/ tf.cast(std_tensor,dtype=tf.float64)
    return image, label


class ConvertData2Hdf5():
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

    def __init__(self, data_dir:Path=None,
                 hdf5_save_dir:Path=None,
                 x_train_arr:np.array=None,
                 y_train_arr:np.array=None,
                 x_test_arr:np.array=None,
                 y_test_arr:np.array=None,
                 data_format:str=None,
                 resizeHeight:int=None,
                 resizeWidth:int=None,
                 augment_bool:bool=None,
                 augmentations_list:list=None,
                 save_augmentation_flag:bool=None,
                 num_classes:int=None
                 ):
        self.logger = logger.get_logger(__name__)
        self.num_classes = num_classes
        self.data_dir = data_dir
        self.data_format= data_format # can take values "np_array", "images"
        self.x_train_arr = x_train_arr
        self.y_train_arr = y_train_arr
        self.x_test_arr = x_test_arr
        self.y_test_arr = y_test_arr
        self.train_dir = self.data_dir/'train'
        self.test_dir = self.data_dir / 'test'
        self.hdf5_save_dir = hdf5_save_dir
        self.hdf5_save_dir.mkdir(parents=True, exist_ok=True)
        self.resizeHeight = resizeHeight
        self.resizeWidth = resizeWidth
        self.augment_bool = augment_bool
        self.augmentations_list = augmentations_list
        self.save_augmentation_flag = save_augmentation_flag
        self.augmentation_func_map = {'random_rotate':random_rotate_90,
                                      'horizonatal_flip': random_flip,
                                      }

    @timeme
    def apply_data_augmentations(self, image:np.array, label:int)->tuple:
        """
        Apply list of augmentations on given image
        """
        aug_images = list()
        for aug in self.augmentations_list:
            assert aug in self.augmentations_list, f'Augmention method not listed in {self.augmentation_func_map.keys()}'
            aug_image = self.augmentation_func_map[aug](image)
            assert len(aug_image.shape) == 3
            aug_images.append(aug_image)
        aug_labels = [label]*len(aug_images)
        return aug_images, aug_labels


    @timeme
    def createHdf5File(self, directory):
        """

        :param directory: Path, can be train or test dir
        :return: None
        """
        hdf5_save_file = self.hdf5_save_dir/f'{directory.name}.h5'
        self.logger.debug(f'Created {hdf5_save_file}')
        hdf5_save_file_obj = tables.open_file(hdf5_save_file.as_posix(), mode='w')

        # create expandable arrays to store images and their labels as arrays
        images_earray = hdf5_save_file_obj.create_earray(where=hdf5_save_file_obj.root,
                                                     name='images',
                                                     atom=tables.UInt8Atom(),
                                                     shape=[0, self.resizeHeight, self.resizeWidth, 3])
        labels_erray = hdf5_save_file_obj.create_earray(where=hdf5_save_file_obj.root,
                                                     name='labels',
                                                     atom=tables.UInt8Atom(),
                                                     shape=[0,1])

        self.logger.debug(f'Raw data is in {self.data_format} format.................................................')
        if self.data_format == 'np_array':
            classes = np.unique(self.y_train_arr)
            self.logger.debug(f'Unique classes identified: {classes}')

            if self.save_augmentation_flag:
                self.logger.debug(f'Data save augmentation enabled, Creating dirs to store augmentations')
                directory_root = directory.parent
                self.logger.debug(f'directory_root: {directory_root.as_posix()}')
                save_aug_dir = directory_root / f'augmented_images'
                save_aug_dir.mkdir(parents=True, exist_ok=True)
                self.logger.debug(f'Creating save_aug_dir: {save_aug_dir.as_posix()}')
                train_aug_dir = save_aug_dir / f'train'
                train_aug_dir.mkdir(parents=True, exist_ok=True)
                self.logger.debug(f'Creating train_aug_dir: {train_aug_dir.as_posix()}')
                # create class wise dirs
                for classlabel in classes:
                    class_dir = train_aug_dir / f'{classlabel}'
                    class_dir.mkdir(parents=True, exist_ok=True)
                    self.logger.debug(f'Creating class_dir: {class_dir.as_posix()}')

            # for training data
            file_name = 0
            for img_arr, label in zip(self.x_train_arr, self.y_train_arr):
                file_name += 1
                label = label[0] # since each label is a list instead of int
                assert len(img_arr.shape) == 3, "Img array from numpy array has more than 3 dim"
                assert type(label) == np.uint8, "Img array label is not an integer"
                if not self.augment_bool:
                    images_earray.append(img_arr[None])
                    labels_erray.append(np.array(label).reshape(-1, 1))
                else:
                    aug_images, aug_labels = self.apply_data_augmentations(img_arr, label)
                    num_augs = 0
                    for aug_image, aug_label in zip(aug_images, aug_labels):
                        num_augs += 1
                        aug_np_image = aug_image.numpy()
                        images_earray.append(aug_np_image[None])
                        labels_erray.append(aug_label)
                        # save aug images
                        if self.save_augmentation_flag:
                            pil_img = Image.fromarray(aug_np_image.astype('uint8'))
                            dest = train_aug_dir / f'{label}/{file_name}_{num_augs}.jpg'
                            self.logger.debug(f'Saving augmented image: {dest.as_posix()}')
                            pil_img.save(dest.as_posix())

            hdf5_save_file_obj.close()

        elif self.data_format == "images":
            classes = [class_path.name for class_path in sorted(list(directory.glob('*')))]
            self.logger.debug(f'Unique classes identified: {classes}')

            # save class index map to hdf5 file
            class_index_df = pd.DataFrame(data={'class':classes,
                               'label_index':list(range(0, len(classes)))
                               }
                         )
            class_index_df.to_hdf(path_or_buf=hdf5_save_file.as_posix(), key='/class_index_df')

            # save augmented images locally if aug is enabled
            if self.save_augmentation_flag:
                directory_root = directory.parent
                self.logger.debug(f'directory_root: {directory_root.as_posix()}')
                save_aug_dir = directory_root / f'augmented_images'
                save_aug_dir.mkdir(parents=True, exist_ok=True)
                self.logger.debug(f'Creating save_aug_dir: {save_aug_dir.as_posix()}')
                train_aug_dir = save_aug_dir / f'train'
                train_aug_dir.mkdir(parents=True, exist_ok=True)
                self.logger.debug(f'Creating train_aug_dir: {train_aug_dir.as_posix()}')
                # create class wise dirs
                for classlabel in class_index_df['class']:
                    class_dir = train_aug_dir / f'{classlabel}'
                    class_dir.mkdir(parents=True, exist_ok=True)
                    self.logger.debug(f'Creating class_dir: {class_dir.as_posix()}')

            for image_path in directory.glob('*/*'):
                image_name = image_path.name
                file_name = image_path.stem
                file_ext = image_path.suffix
                class_name = image_path.parent.name
                class_index = class_index_df[class_index_df['class']==class_name]['label_index'].values[0]

                # read image and resize it
                img = np.array(Image.open(image_path))
                rs_img = resize_image(img_tensor=tf.convert_to_tensor(img),
                                      resize_shape=(self.resizeHeight, self.resizeWidth)
                                      ).numpy()

                if not self.augment_bool:
                    # add image and label to hdf5 earray
                    images_earray.append(rs_img)
                    labels_erray.append(np.array(class_index).reshape(-1, 1))
                else:
                    aug_images, aug_labels = self.apply_data_augmentations(rs_img, class_index)
                    num_augs = 0
                    for aug_image, aug_label in zip(aug_images, aug_labels):
                        num_augs += 1
                        aug_np_image = aug_image.numpy()
                        images_earray.append(aug_np_image[None])
                        labels_erray.append(np.array(aug_label).reshape(-1, 1))

                        # save aug images
                        if self.save_augmentation_flag:
                            pil_img = Image.fromarray(aug_np_image.astype('uint8'))
                            dest = train_aug_dir/f'{class_name}/{file_name}_{num_augs}{file_ext}'
                            self.logger.debug(f'Saving augmented image: {dest.as_posix()}')
                            pil_img.save(dest.as_posix())

                    # add original image
                    images_earray.append(rs_img[None])
                    labels_erray.append(np.array(class_index).reshape(-1, 1))


            hdf5_save_file_obj.close()

    @timeme
    def createTrainTestHdf5Files(self):
        self.createHdf5File(directory=self.train_dir)
        self.createHdf5File(directory=self.test_dir)


# preprocessing methods
@timeme
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

@timeme
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

@timeme
def random_rotate_90(x: tf.Tensor) -> tf.Tensor:
    """
    Randomly rotate the input image by either 0, 90, 180 or 270 degrees.
    :param x: Input image.
    :return: Transformed image.
    """
    return tf.image.rot90(x, tf.random.uniform(shape=[], minval=10, maxval=15, dtype=tf.int32))

def deg2rad(x: tf.Tensor) -> tf.Tensor:
    """
    Converts an angle in degrees to radians.
    :param x: Input angle, in degrees.
    :return: Angle in radians
    """
    return (x * np.pi) / 180


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

if __name__ == "__main__":
    # (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
    # o = ConvertData2Hdf5(data_dir=Path("/Users/satishjasthi/Downloads/raw_data/"),
    #                      hdf5_save_dir=Path("/Users/satishjasthi/Downloads/raw_data/HDF5"),
    #                      x_train_arr=x_train,
    #                      y_train_arr=y_train,
    #                      x_test_arr=x_test,
    #                      y_test_arr=y_test,
    #                      data_format="np_array",
    #                      resizeHeight=32,
    #                      resizeWidth=32,
    #                      augment_bool=True,
    #                      augmentations_list=['random_rotate', 'horizonatal_flip'],
    #                      save_augmentation_flag=True,
    #                                )
    # o.createTrainTestHdf5Files()
    o = TFRecords(mode='train', TrainHdf5_data=Path(r"C:\Users\neere\Desktop\deleteme\raw_dir\HDF5Data\train.h5"),
                  TrainTfRecord_data=Path(r"C:\Users\neere\Desktop\deleteme\raw_dir\TFRecords\train.tfrecords"))
    # o.writeTfRecord()
    raw_data = o.readTfRecord()
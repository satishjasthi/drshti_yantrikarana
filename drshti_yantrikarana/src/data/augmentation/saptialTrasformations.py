"""
Reference: 
Usage:
TF_version:2.0

About: Script to create data augmentations that involve spatial transformations like
    - random rotate
    -  random flip
# TODO add all augmentation methods here
Author: Satish Jasthi
"""
import random

import tensorflow as tf
import numpy as np
from PIL import Image

from drshti_yantrikarana.config import rotation_min, rotation_max


def convertNpArray2Image(arr: np.array) -> Image:
    """
    Function to convert np array to PIL image
    """
    return Image.fromarray(arr.astype('uint8'))


def random_flip(image:np.array, flip_mode: str) -> Image:
    """
    Function to randlomy flip images with probability of 0.5
    flip_mode:
        - 'h' for horizontal flip
        - 'v' for vertical flip
    """

    if flip_mode == 'h':
        flpd_img = tf.image.random_flip_left_right(image)
    elif flip_mode == 'v':
        flpd_img = tf.image.random_flip_up_down(image)
    else:
        raise NotImplementedError(f'current flip mode: {flip_mode} does not belongs to defined flip modes(h, w)')
    return flpd_img.numpy()


def pad_image(image:np.array, padding: list) -> Image:
    """
    Function to pad an image
    padding: For each dimension D of input, paddings[D, 0] indicates how many values to add before the contents of
    tensor in that dimension, and paddings[D, 1] indicates how many values to add after the contents of tensor in that
    dimension.
    """
    image = np.array(image)
    padding = tf.constant(padding)
    padded_img = tf.pad(image, padding, mode='CONSTANT')
    return padded_img.numpy()


def random_crop(image:np.array, height: int, width: int, depth: int) -> Image:
    """
    Function to randomly crop an image to dim (height, width)
    """
    image = np.array(image)
    crp_img = tf.image.random_crop(image, size=(height, width, depth))
    return crp_img.numpy()


def equalize(image:np.array) -> Image:
    """Implements Equalize function from PIL using TF ops.
  to understand equalization see: https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_histograms/py_histogram_equalization/py_histogram_equalization.html
  """

    image = tf.convert_to_tensor(np.array(image))

    def scale_channel(im: np.array, c: int):
        """Scale the data in the channel to implement equalize."""
        im = tf.cast(im[:, :, c], tf.int32)
        # Compute the histogram of the image channel.
        histo = tf.histogram_fixed_width(im, [0, 255], nbins=256)

        # For the purposes of computing the step, filter out the nonzeros.
        nonzero = tf.where(tf.not_equal(histo, 0))
        nonzero_histo = tf.reshape(tf.gather(histo, nonzero), [-1])
        step = (tf.reduce_sum(nonzero_histo) - nonzero_histo[-1]) // 255

        def build_lut(histo, step):
            # Compute the cumulative sum, shifting by step // 2
            # and then normalization by step.
            lut = (tf.cumsum(histo) + (step // 2)) // step
            # Shift lut, prepending with 0.
            lut = tf.concat([[0], lut[:-1]], 0)
            # Clip the counts to be in range.  This is done
            # in the C code for image.point.
            return tf.clip_by_value(lut, 0, 255)

        # If step is zero, return the original image.  Otherwise, build
        # lut from the full histogram and step and then index from it.
        result = tf.cond(tf.equal(step, 0),
                         lambda: im,
                         lambda: tf.gather(build_lut(histo, step), im))

        return tf.cast(result, tf.uint8)

    # Assumes RGB for now.  Scales each channel independently
    # and then stacks the result.
    s1 = scale_channel(image, 0)
    s2 = scale_channel(image, 1)
    s3 = scale_channel(image, 2)
    image = tf.stack([s1, s2, s3], 2)
    return image.numpy()


def autocontrast(image:np.array) -> Image:
    """Implements Autocontrast function from PIL using TF ops.
  """
    image = tf.convert_to_tensor(np.array(image))
    def scale_channel(image: np.array):
        """Scale the 2D image using the autocontrast rule."""
        # A possibly cheaper version can be done using cumsum/unique_with_counts
        # over the histogram values, rather than iterating over the entire image.
        # to compute mins and maxes.
        lo = tf.cast(tf.reduce_min(image), dtype=tf.float32)
        hi = tf.cast(tf.reduce_max(image), dtype=tf.float32)

        # Scale the image, making the lowest value 0 and the highest value 255.
        def scale_values(im):
            scale = 255.0 / (hi - lo)
            offset = -lo * scale
            im = tf.cast(im, dtype=tf.float32) * scale + offset
            im = tf.clip_by_value(im, 0.0, 255.0)
            return tf.cast(im, tf.uint8)

        result = tf.cond(hi > lo, lambda: scale_values(image), lambda: image)
        return result

    # Assumes RGB for now.  Scales each channel independently
    # and then stacks the result.
    s1 = scale_channel(image[:, :, 0])
    s2 = scale_channel(image[:, :, 1])
    s3 = scale_channel(image[:, :, 2])
    image = tf.stack([s1, s2, s3], 2)
    return image.numpy()


def blend(image1: np.array, image2: np.array, factor: float = 0.5):
    """Blend image1 and image2 using 'factor'.
  Factor can be above 0.0.  A value of 0.0 means only image1 is used.
  A value of 1.0 means only image2 is used.  A value between 0.0 and
  1.0 means we linearly interpolate the pixel values between the two
  images.  A value greater than 1.0 "extrapolates" the difference
  between the two pixel values, and we clip the results to values
  between 0 and 255.
  Args:
    image1: An image Tensor of type uint8.
    image2: An image Tensor of type uint8.
    factor: A floating point value above 0.0.
  Returns:
    A blended image Tensor of type uint8.
  """
    if factor == 0.0:
        return tf.convert_to_tensor(image1)
    if factor == 1.0:
        return tf.convert_to_tensor(image2)

    image1 = tf.cast(image1, dtype=tf.float32)
    image2 = tf.cast(image2, dtype=tf.float32)

    difference = image2 - image1
    scaled = factor * difference

    # Do addition in float.
    temp = tf.cast(image1, dtype=tf.float32) + scaled

    # Interpolate
    if factor > 0.0 and factor < 1.0:
        # Interpolation means we always stay within 0 and 255.
        return tf.cast(temp, tf.uint8)

    # Extrapolate:
    #
    # We need to clip and then cast.
    return tf.cast(tf.clip_by_value(temp, 0.0, 255.0), tf.uint8).numpy()


def color(image:np.array, factor: float = 0.6) -> Image:
    """Equivalent of PIL Color.
    factor is 0.6 by averaging different factors used in auto augment paper
    """
    image = np.array(image)
    degenerate = tf.image.grayscale_to_rgb(tf.image.rgb_to_grayscale(image))
    img = blend(degenerate, image, factor)
    return img.numpy()


def brightness(image:np.array, factor: float = 0.5) -> Image:
    """Equivalent of PIL Brightness.
    factor is 0.5 by averaging different factors used in auto augment paper
    """
    image = np.array(image)
    degenerate = tf.zeros_like(image)
    img = blend(degenerate, image, factor)
    return img.numpy()


def apply_randomRotation(image:np.array) -> Image:
    """
    Function to rotate image by random degree value
    """
    image_arr = np.array(image)
    image = Image.fromarray(image_arr)
    rand_angle = np.random.randint(rotation_min, rotation_max)
    image_rot = Image.Image.rotate(image, angle=rand_angle)
    return np.array(image_rot)


def apply_flip(image: tf.image, mode='h') -> tf.image:
    """
    Function to flip an image either vertically or horizontally
    :param image:tf.image
    :param mode: str, 'v' for vertical flip and 'h' for horizontal flip
    :return: tf.tensor
    """
    if mode == 'v':
        return tf.image.flip_up_down(image).numpy()
    elif mode == 'h':
        return tf.image.flip_left_right(image).numpy()

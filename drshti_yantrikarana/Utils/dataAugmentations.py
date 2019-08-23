"""
Reference: 
Usage:

About:

Author: Satish Jasthi
"""
import numpy as np
import tensorflow as tf


def random_flip(x: np.array, flip_vert: bool = False) -> np.array:
    """
    Randomly flip the input image horizontally, and optionally also vertically, which is implemented as 90-degree
    rotations.
    :param x: np.array.
    :param flip_vert: Whether to perform vertical flipping. Default: False.
    :return: np.array
    """
    x = tf.convert_to_tensor(x)
    x = tf.image.random_flip_left_right(x)
    if flip_vert:
        x = random_rotate_90(x)
    return x.numpy()


def random_rotate_90(x: np.array) -> np.array:
    """
    Randomly rotate the input image by either 0, 90, 180 or 270 degrees.
    :param x: np.array.
    :return: np.array
    """
    x = tf.convert_to_tensor(x)
    return tf.image.rot90(x, tf.random.uniform(shape=[], minval=10, maxval=15, dtype=tf.int32)).numpy()

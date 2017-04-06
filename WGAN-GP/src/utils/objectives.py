import tensorflow as tf


def mae(pred, target, name='mae'):

    return tf.reduce_mean(tf.abs(pred - target), name=name)


def mse(pred, target, name='mse'):

    return tf.reduce_mean(tf.square(pred - target), name=name)


def pixel_rmse(pred, target, name='rmse'):

    return tf.sqrt(tf.reduce_mean(tf.square(pred - target), name=name))


def binary_cross_entropy_with_logits(pred, target):

    return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=pred, labels=target))


def wasserstein(pred, target):

    return tf.reduce_mean(pred * target)

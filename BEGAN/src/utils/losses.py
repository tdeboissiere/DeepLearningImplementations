import tensorflow as tf


def mae(pred, target, name='mae'):

    return tf.reduce_mean(tf.abs(pred - target), name=name)


def mse(pred, target, name='mse'):

    return tf.reduce_mean(tf.square(pred - target), name=name)


def rmse(pred, target, name='rmse'):

    return tf.sqrt(tf.reduce_mean(tf.square(pred - target), name=name))


def binary_cross_entropy_with_logits(pred, target):

    return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=pred, labels=target))


def wasserstein(pred, target):

    return tf.reduce_mean(pred * target)


def sparse_cross_entropy(pred, target):

    return tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=pred, labels=target))


def cross_entropy(pred, target):

    return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=target))


def KL(mu, logsigma):
    # KL Divergence between latents and Standard Normal prior
    kl_div = -0.5 * tf.reduce_mean(1 + 2 * logsigma - tf.sqrt(mu) - tf.exp(2 * logsigma))

    return kl_div

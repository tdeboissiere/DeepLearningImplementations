import os
# Disable Tensorflow's INFO and WARNING messages
# See http://stackoverflow.com/questions/35911252
if 'TF_CPP_MIN_LOG_LEVEL' not in os.environ:
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import flags
import train

FLAGS = tf.app.flags.FLAGS


def launch_training():

    train.train_model()


def main(argv=None):

    assert FLAGS.run in ["train"], "Choose [train]"

    if FLAGS.run == 'train':
        launch_training()


if __name__ == '__main__':

    flags.define_flags()
    tf.app.run()

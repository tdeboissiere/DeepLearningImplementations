import os
# Disable Tensorflow's INFO and WARNING messages
# See http://stackoverflow.com/questions/35911252
if 'TF_CPP_MIN_LOG_LEVEL' not in os.environ:
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import flags
import train
# INFERENCE TODO
# import inference

FLAGS = tf.app.flags.FLAGS


def launch_training():

    train.train_model()


# def launch_inference():

#     inference.infer()


def main(argv=None):

    # INFERENCE TODO
    assert FLAGS.run in ["train"], "Choose [train]"

    if FLAGS.run == 'train':
        launch_training()

    # if FLAGS.run == 'inference':
    #     launch_inference()


if __name__ == '__main__':

    flags.define_flags()
    tf.app.run()

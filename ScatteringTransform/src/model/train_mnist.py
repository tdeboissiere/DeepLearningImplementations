import sys
import models
import numpy as np
from tqdm import tqdm
import tensorflow as tf
sys.path.append("../utils")
import training_utils as tu
import data_utils as du
import layers
import scattering

from tensorflow.examples.tutorials.mnist import input_data

FLAGS = tf.app.flags.FLAGS


def train_model():

    # Setup session
    sess = tu.setup_session()

    # Placeholder for data and Mnist iterator
    mnist = input_data.read_data_sets(FLAGS.raw_dir, one_hot=True)
    assert FLAGS.data_format == "NCHW", "Scattering only implemented in NCHW"
    X_tensor = tf.placeholder(tf.float32, shape=[FLAGS.batch_size, 1, 28, 28])
    y_tensor = tf.placeholder(tf.int64, shape=[FLAGS.batch_size, 10])

    with tf.device('/cpu:0'):
        X_train = mnist.train.images.astype(np.float32)
        y_train = mnist.train.labels.astype(np.int64)

        X_validation = mnist.validation.images.astype(np.float32)
        y_validation = mnist.validation.labels.astype(np.int64)

        X_train = (X_train - 0.5) / 0.5
        X_train = X_train.reshape((-1, 1, 28, 28))

        X_validation = (X_validation - 0.5) / 0.5
        X_validation = X_validation.reshape((-1, 1, 28, 28))

    # Build model
    class HybridCNN(models.Model):

        def __call__(self, x, reuse=False):
            with tf.variable_scope(self.name) as scope:

                if reuse:
                    scope.reuse_variables()

                M, N = x.get_shape().as_list()[-2:]
                x = scattering.Scattering(M=M, N=N, J=2)(x)
                x = tf.contrib.layers.batch_norm(x, data_format=FLAGS.data_format, fused=True, scope="scat_bn")
                x = layers.conv2d_block("CONV2D", x, 64, 1, 1, p="SAME", data_format=FLAGS.data_format, bias=True, bn=False, activation_fn=tf.nn.relu)

                target_shape = (-1, 64 * 7 * 7)
                x = layers.reshape(x, target_shape)
                x = layers.linear(x, 512, name="dense1")
                x = tf.nn.relu(x)
                x = layers.linear(x, 10, name="dense2")

                return x

    HCNN = HybridCNN("HCNN")
    y_pred = HCNN(X_tensor)

    ###########################
    # Instantiate optimizers
    ###########################
    opt = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate, name='opt', beta1=0.5)

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_tensor, logits=y_pred))
    correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y_tensor, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    ###########################
    # Compute gradient updates
    ###########################

    dict_vars = HCNN.get_trainable_variables()
    all_vars = [dict_vars[k] for k in dict_vars.keys()]

    gradvar = opt.compute_gradients(loss, var_list=all_vars, colocate_gradients_with_ops=True)
    update = opt.apply_gradients(gradvar, name='loss_minimize')

    ##########################
    # Group training ops
    ##########################
    train_ops = [update]
    loss_ops = [loss, accuracy]

    ##########################
    # Summary ops
    ##########################

    # Add summary for gradients
    tu.add_gradient_summary(gradvar)

    # Add scalar symmaries
    tf.summary.scalar("loss", loss)

    summary_op = tf.summary.merge_all()

    ############################
    # Start training
    ############################

    # Initialize session
    tu.initialize_session(sess)

    # Start queues
    tu.manage_queues(sess)

    # Summaries
    writer = tu.manage_summaries(sess)

    for e in tqdm(range(FLAGS.nb_epoch), desc="Training progress"):

        t = tqdm(range(FLAGS.nb_batch_per_epoch), desc="Epoch %i" % e, mininterval=0.5)
        for batch_counter in t:

            # Get training data
            X_train_batch, y_train_batch = du.sample_batch(X_train, y_train, FLAGS.batch_size)

            # Run update and get loss
            output = sess.run(train_ops + loss_ops + [summary_op], feed_dict={X_tensor: X_train_batch,
                                                                              y_tensor: y_train_batch})
            train_loss = output[1]
            train_acc = output[2]

            # Write summaries
            if batch_counter % (FLAGS.nb_batch_per_epoch // 20) == 0:
                writer.add_summary(output[-1], e * FLAGS.nb_batch_per_epoch + batch_counter)

            # Get validation data
            X_validation_batch, y_validation_batch = du.sample_batch(X_validation, y_validation, FLAGS.batch_size)

            # Run update and get loss
            output = sess.run(loss_ops, feed_dict={X_tensor: X_validation_batch,
                                                   y_tensor: y_validation_batch})
            validation_loss = output[0]
            validation_acc = output[1]

            t.set_description('Epoch %i: - train loss: %.2f val loss: %.2f - train acc: %.2f val acc: %.2f' %
                              (e, train_loss, validation_loss, train_acc, validation_acc))

    print('Finished training!')

import os
import sys
from tqdm import tqdm
import tensorflow as tf
import models
import numpy as np
sys.path.append("../utils")
import losses
import data_utils as du
import training_utils as tu
import visualization_utils as vu

FLAGS = tf.app.flags.FLAGS


def train_model():

    # Setup session
    sess = tu.setup_training_session()

    ##########
    # Innputs
    ##########

    # Setup async input queue of real images
    X_real = du.read_celebA()

    # Noise
    batch_size = tf.shape(X_real)[0]
    z_noise = tf.random_uniform((batch_size, FLAGS.z_dim), minval=-1, maxval=1, name="z_input")
    epsilon = tf.random_uniform((batch_size, 1, 1, 1), minval=0, maxval=1, name="epsilon")

    # learning rate
    lr_D = tf.Variable(initial_value=FLAGS.learning_rate, trainable=False, name='learning_rate')
    lr_G = tf.Variable(initial_value=FLAGS.learning_rate, trainable=False, name='learning_rate')

    ########################
    # Instantiate models
    ########################
    G = models.Generator()
    D = models.Discriminator()

    ###########################
    # Instantiate optimizers
    ###########################
    G_opt = tf.train.AdamOptimizer(learning_rate=lr_D, name='G_opt', beta1=0.5)
    D_opt = tf.train.AdamOptimizer(learning_rate=lr_G, name='D_opt', beta1=0.5)

    ##########
    # Outputs
    ##########
    X_fake = G(z_noise)
    X_hat = epsilon * X_real + (1 - epsilon) * X_fake

    D_real = D(X_real)
    D_fake = D(X_fake, reuse=True)
    D_X_hat = D(X_hat, reuse=True)

    grad_D_X_hat = tf.gradients(D_X_hat, X_hat)[0]

    # output images
    generated_toplot = du.unnormalize_image(X_fake, name="generated_toplot")
    real_toplot = du.unnormalize_image(X_real, name="real_toplot")

    ###########################
    # losses
    ###########################

    G_loss = losses.wasserstein(D_fake, -tf.ones_like(D_fake))
    D_loss_grad = FLAGS.lbd * tf.square((tf.nn.l2_loss(grad_D_X_hat) - 1))
    D_loss_real = losses.wasserstein(D_real, -tf.ones_like(D_real))
    D_loss_fake = losses.wasserstein(D_fake, tf.ones_like(D_fake))
    D_loss = D_loss_grad + D_loss_real + D_loss_fake

    ###########################
    # Compute updates ops
    ###########################

    dict_G_vars = G.get_trainable_variables()
    G_vars = [dict_G_vars[k] for k in dict_G_vars.keys()]

    dict_D_vars = D.get_trainable_variables()
    D_vars = [dict_D_vars[k] for k in dict_D_vars.keys()]

    G_gradvar = G_opt.compute_gradients(G_loss, var_list=G_vars)
    G_update = G_opt.apply_gradients(G_gradvar, name='G_loss_minimize')

    D_gradvar = D_opt.compute_gradients(D_loss, var_list=D_vars)
    D_update = D_opt.apply_gradients(D_gradvar, name='D_loss_minimize')

    # D_gradvar_fake = D_opt.compute_gradients(D_loss_fake, var_list=D_vars)
    # D_update_fake = D_opt.apply_gradients(D_gradvar_fake, name='D_loss_minimize_fake')

    ##########################
    # Summary ops
    ##########################

    # Add summary for gradients
    tu.add_gradient_summary(G_gradvar)
    tu.add_gradient_summary(D_gradvar)
    # tu.add_gradient_summary(D_gradvar_fake)

    # Add scalar symmaries for G
    tf.summary.scalar("G loss", G_loss)
    # Add scalar symmaries for D
    tf.summary.scalar("D loss real", D_loss_real)
    tf.summary.scalar("D loss fake", D_loss_fake)
    tf.summary.scalar("D loss grad", D_loss_grad)
    # Add scalar symmaries for D

    summary_op = tf.summary.merge_all()

    ############################
    # Start training
    ############################

    # Initialize session
    saver = tu.initialize_session(sess)

    # Start queues
    coord, threads = du.manage_queues(sess)

    # Summaries
    writer = tu.manage_summaries(sess)

    # Run checks on data dimensions
    list_data = [z_noise]
    list_data += [X_real, X_fake]
    list_data += [generated_toplot, real_toplot]
    output = sess.run(list_data)
    tu.check_data(output, list_data)

    for e in tqdm(range(FLAGS.nb_epoch), desc="Training progress"):

        t = tqdm(range(FLAGS.nb_batch_per_epoch), desc="Epoch %i" % e, mininterval=0.5)
        for batch_counter in t:

            # Update discriminator
            for i_D in range(FLAGS.ncritic):
                sess.run([D_update])
                # r = np.random.randint(0, 2)
                # if r == 0:
                #     sess.run([D_update_real])
                # else:
                #     sess.run([D_update_fake])

            # Update generator
            sess.run([G_update])

            if batch_counter % (FLAGS.nb_batch_per_epoch // (int(0.5 * FLAGS.nb_batch_per_epoch))) == 0:
                output = sess.run([summary_op])
                writer.add_summary(output[-1], e * FLAGS.nb_batch_per_epoch + batch_counter)

            t.set_description('Epoch %s:' % e)

        # Plot some generated images
        Xf, Xr = sess.run([generated_toplot, real_toplot])
        vu.save_image(Xf, Xr, title="current_batch", e=e)

        # Save session
        saver.save(sess, os.path.join(FLAGS.model_dir, "model"), global_step=e)

        # Show data statistics
        output = sess.run(list_data)
        tu.check_data(output, list_data)

    # Stop threads
    coord.request_stop()
    coord.join(threads)

    print('Finished training!')

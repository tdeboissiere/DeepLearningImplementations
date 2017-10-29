import os
import sys
from tqdm import tqdm
import tensorflow as tf
import models
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
    z_noise_for_D = tf.random_uniform((batch_size, FLAGS.z_dim,), minval=-1, maxval=1, name="z_input_D")
    z_noise_for_G = tf.random_uniform((batch_size, FLAGS.z_dim,), minval=-1, maxval=1, name="z_input_G")

    # k factor
    k_factor = tf.Variable(initial_value=0., trainable=False, name='anneal_factor')

    # learning rate
    lr = tf.Variable(initial_value=FLAGS.learning_rate, trainable=False, name='learning_rate')

    ########################
    # Instantiate models
    ########################
    G = models.Generator(nb_filters=FLAGS.nb_filters_G)
    D = models.Discriminator(h_dim=FLAGS.h_dim, nb_filters=FLAGS.nb_filters_D)

    ##########
    # Outputs
    ##########
    X_rec_real = D(X_real, output_name="X_rec_real")

    X_fake_for_D = G(z_noise_for_D, output_name="X_fake_for_D")
    X_rec_fake_for_D = D(X_fake_for_D, reuse=True, output_name="X_rec_fake_for_D")

    X_fake_for_G = G(z_noise_for_G, reuse=True, output_name="X_fake_for_G")
    X_rec_fake_for_G = D(X_fake_for_G, reuse=True, output_name="X_rec_fake_for_G")

    # output images for plots
    real_toplot = du.unnormalize_image(X_real, name="real_toplot")
    generated_toplot = du.unnormalize_image(X_fake_for_G, name="generated_toplot")
    real_rec_toplot = du.unnormalize_image(X_rec_real, name="rec_toplot")
    generated_rec_toplot = du.unnormalize_image(X_rec_fake_for_G, name="generated_rec_toplot")

    ###########################
    # Instantiate optimizers
    ###########################
    opt = tf.train.AdamOptimizer(learning_rate=lr, name='opt')

    ###########################
    # losses
    ###########################

    loss_real = losses.mae(X_real, X_rec_real)
    loss_fake_for_D = losses.mae(X_fake_for_D, X_rec_fake_for_D)
    loss_fake_for_G = losses.mae(X_fake_for_G, X_rec_fake_for_G)

    L_D = loss_real - k_factor * loss_fake_for_D
    L_G = loss_fake_for_G
    Convergence = loss_real + tf.abs(FLAGS.gamma * loss_real - loss_fake_for_G)

    ###########################
    # Compute updates ops
    ###########################

    dict_G_vars = G.get_trainable_variables()
    G_vars = [dict_G_vars[k] for k in dict_G_vars.keys()]

    dict_D_vars = D.get_trainable_variables()
    D_vars = [dict_D_vars[k] for k in dict_D_vars.keys()]

    G_gradvar = opt.compute_gradients(L_G, var_list=G_vars)
    G_update = opt.apply_gradients(G_gradvar, name='G_loss_minimize')

    D_gradvar = opt.compute_gradients(L_D, var_list=D_vars)
    D_update = opt.apply_gradients(D_gradvar, name='D_loss_minimize')

    update_k_factor = tf.assign(k_factor, k_factor + FLAGS.lambdak * (FLAGS.gamma * loss_real - loss_fake_for_G))
    update_lr = tf.assign(lr, tf.maximum(1E-6, lr / 2))

    ##########################
    # Summary ops
    ##########################

    # Add summary for gradients
    tu.add_gradient_summary(G_gradvar)
    tu.add_gradient_summary(D_gradvar)

    # Add scalar symmaries for G
    tf.summary.scalar("G loss", L_G)
    # Add scalar symmaries for D
    tf.summary.scalar("D loss", L_D)
    # Add scalar symmaries for D
    tf.summary.scalar("k_factor", k_factor)
    tf.summary.scalar("Convergence", Convergence)
    tf.summary.scalar("learning rate", lr)

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
    list_data = [z_noise_for_D, z_noise_for_G]
    list_data += [X_real, X_rec_real, X_fake_for_G, X_rec_fake_for_G, X_fake_for_D, X_rec_fake_for_D]
    list_data += [generated_toplot, real_toplot]
    output = sess.run(list_data)
    tu.check_data(output, list_data)

    for e in tqdm(range(FLAGS.nb_epoch), desc="Training progress"):

        # Anneal learning rate
        if (e + 1) % 200 == 0:
            sess.run([update_lr])

        t = tqdm(range(FLAGS.nb_batch_per_epoch), desc="Epoch %i" % e, mininterval=0.5)
        for batch_counter in t:

            output = sess.run([G_update, D_update, update_k_factor])

            if batch_counter % (FLAGS.nb_batch_per_epoch // (int(0.5 * FLAGS.nb_batch_per_epoch))) == 0:
                output = sess.run([summary_op])
                writer.add_summary(output[-1], e * FLAGS.nb_batch_per_epoch + batch_counter)

        # Plot some generated images
        Xf, Xr, Xrrec, Xfrec = sess.run([generated_toplot, real_toplot, real_rec_toplot, generated_rec_toplot])
        vu.save_image(Xf, Xr, title="current_batch", e=e)
        vu.save_image(Xrrec, Xfrec, title="reconstruction", e=e)

        # Save session
        saver.save(sess, os.path.join(FLAGS.model_dir, "model"), global_step=e)

        # Show data statistics
        output = sess.run(list_data)
        tu.check_data(output, list_data)

    # Stop threads
    coord.request_stop()
    coord.join(threads)

    print('Finished training!')

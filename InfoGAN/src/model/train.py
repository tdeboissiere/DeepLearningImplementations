import os
import sys
import time
import models as models
from keras.utils import generic_utils
from keras.optimizers import Adam, SGD
import keras.backend as K
# Utils
sys.path.append("../utils")
import general_utils
import data_utils


def gaussian_loss(y_true, y_pred):

    Q_C_mean = y_pred[:, 0, :]
    Q_C_logstd = y_pred[:, 1, :]

    y_true = y_true[:, 0, :]

    epsilon = (y_true - Q_C_mean) / (K.exp(Q_C_logstd) + K.epsilon())
    loss_Q_C = (Q_C_logstd + 0.5 * K.square(epsilon))
    loss_Q_C = K.mean(loss_Q_C)

    return loss_Q_C


def train(**kwargs):
    """
    Train model

    Load the whole train data in memory for faster operations

    args: **kwargs (dict) keyword arguments that specify the model hyperparameters
    """

    # Roll out the parameters
    batch_size = kwargs["batch_size"]
    n_batch_per_epoch = kwargs["n_batch_per_epoch"]
    nb_epoch = kwargs["nb_epoch"]
    generator = kwargs["generator"]
    model_name = kwargs["model_name"]
    image_dim_ordering = kwargs["image_dim_ordering"]
    img_dim = kwargs["img_dim"]
    cont_dim = (kwargs["cont_dim"],)
    cat_dim = (kwargs["cat_dim"],)
    noise_dim = (kwargs["noise_dim"],)
    bn_mode = kwargs["bn_mode"]
    label_smoothing = kwargs["label_smoothing"]
    label_flipping = kwargs["label_flipping"]
    noise_scale = kwargs["noise_scale"]
    dset = kwargs["dset"]
    use_mbd = kwargs["use_mbd"]
    epoch_size = n_batch_per_epoch * batch_size

    # Setup environment (logging directory etc)
    general_utils.setup_logging(model_name)

    # Load and rescale data
    if dset == "celebA":
        X_real_train = data_utils.load_celebA(img_dim, image_dim_ordering)
    if dset == "mnist":
        X_real_train, _, _, _ = data_utils.load_mnist(image_dim_ordering)
    img_dim = X_real_train.shape[-3:]

    try:

        # Create optimizers
        opt_dcgan = Adam(lr=1E-4, beta_1=0.5, beta_2=0.999, epsilon=1e-08)
        opt_discriminator = Adam(lr=1E-4, beta_1=0.5, beta_2=0.999, epsilon=1e-08)
        # opt_discriminator = SGD(lr=1E-4, momentum=0.9, nesterov=True)

        # Load generator model
        generator_model = models.load("generator_%s" % generator,
                                      cat_dim,
                                      cont_dim,
                                      noise_dim,
                                      img_dim,
                                      bn_mode,
                                      batch_size,
                                      dset=dset,
                                      use_mbd=use_mbd)
        # Load discriminator model
        discriminator_model = models.load("DCGAN_discriminator",
                                          cat_dim,
                                          cont_dim,
                                          noise_dim,
                                          img_dim,
                                          bn_mode,
                                          batch_size,
                                          dset=dset,
                                          use_mbd=use_mbd)

        generator_model.compile(loss='mse', optimizer=opt_discriminator)
        discriminator_model.trainable = False

        DCGAN_model = models.DCGAN(generator_model,
                                   discriminator_model,
                                   cat_dim,
                                   cont_dim,
                                   noise_dim)

        list_losses = ['binary_crossentropy', 'categorical_crossentropy', gaussian_loss]
        list_weights = [1, 1, 1]
        DCGAN_model.compile(loss=list_losses, loss_weights=list_weights, optimizer=opt_dcgan)

        # Multiple discriminator losses
        discriminator_model.trainable = True
        discriminator_model.compile(loss=list_losses, loss_weights=list_weights, optimizer=opt_discriminator)

        gen_loss = 100
        disc_loss = 100

        # Start training
        print("Start training")
        for e in range(nb_epoch):
            # Initialize progbar and batch counter
            progbar = generic_utils.Progbar(epoch_size)
            batch_counter = 1
            start = time.time()

            for X_real_batch in data_utils.gen_batch(X_real_train, batch_size):

                # Create a batch to feed the discriminator model
                X_disc, y_disc, y_cat, y_cont = data_utils.get_disc_batch(X_real_batch,
                                                                          generator_model,
                                                                          batch_counter,
                                                                          batch_size,
                                                                          cat_dim,
                                                                          cont_dim,
                                                                          noise_dim,
                                                                          noise_scale=noise_scale,
                                                                          label_smoothing=label_smoothing,
                                                                          label_flipping=label_flipping)

                # Update the discriminator
                disc_loss = discriminator_model.train_on_batch(X_disc, [y_disc, y_cat, y_cont])

                # Create a batch to feed the generator model
                X_gen, y_gen, y_cat, y_cont, y_cont_target = data_utils.get_gen_batch(batch_size,
                                                                                      cat_dim,
                                                                                      cont_dim,
                                                                                      noise_dim,
                                                                                      noise_scale=noise_scale)

                # Freeze the discriminator
                discriminator_model.trainable = False
                gen_loss = DCGAN_model.train_on_batch([y_cat, y_cont, X_gen], [y_gen, y_cat, y_cont_target])
                # Unfreeze the discriminator
                discriminator_model.trainable = True

                batch_counter += 1
                progbar.add(batch_size, values=[("D tot", disc_loss[0]),
                                                ("D log", disc_loss[1]),
                                                ("D cat", disc_loss[2]),
                                                ("D cont", disc_loss[3]),
                                                ("G tot", gen_loss[0]),
                                                ("G log", gen_loss[1]),
                                                ("G cat", gen_loss[2]),
                                                ("G cont", gen_loss[3])])

                # Save images for visualization
                if batch_counter % (n_batch_per_epoch / 2) == 0:
                    data_utils.plot_generated_batch(X_real_batch, generator_model,
                                                    batch_size, cat_dim, cont_dim, noise_dim, image_dim_ordering)

                if batch_counter >= n_batch_per_epoch:
                    break

            print("")
            print('Epoch %s/%s, Time: %s' % (e + 1, nb_epoch, time.time() - start))

            if e % 5 == 0:
                gen_weights_path = os.path.join('../../models/%s/gen_weights_epoch%s.h5' % (model_name, e))
                generator_model.save_weights(gen_weights_path, overwrite=True)

                disc_weights_path = os.path.join('../../models/%s/disc_weights_epoch%s.h5' % (model_name, e))
                discriminator_model.save_weights(disc_weights_path, overwrite=True)

                DCGAN_weights_path = os.path.join('../../models/%s/DCGAN_weights_epoch%s.h5' % (model_name, e))
                DCGAN_model.save_weights(DCGAN_weights_path, overwrite=True)

    except KeyboardInterrupt:
        pass

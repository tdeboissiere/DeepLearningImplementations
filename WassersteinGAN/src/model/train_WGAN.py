import os
import sys
import time
import numpy as np
import models_WGAN as models
from keras.utils import generic_utils
from keras.optimizers import Adam, SGD, RMSprop
# Utils
sys.path.append("../utils")
import general_utils
import data_utils


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
    bn_mode = kwargs["bn_mode"]
    noise_scale = kwargs["noise_scale"]
    dset = kwargs["dset"]
    epoch_size = n_batch_per_epoch * batch_size

    # Setup environment (logging directory etc)
    general_utils.setup_logging(model_name)

    # Load and rescale data
    if dset == "celebA":
        X_real_train = data_utils.load_celebA(img_dim, image_dim_ordering)
    if dset == "mnist":
        X_real_train, _, _, _ = data_utils.load_mnist(image_dim_ordering)
    if dset == "cifar10":
        X_real_train, _, _, _ = data_utils.load_cifar10(image_dim_ordering)
    img_dim = X_real_train.shape[-3:]
    noise_dim = (100,)

    try:

        # Create optimizers
        opt_dcgan = RMSprop(lr=5E-5)
        opt_discriminator = RMSprop(lr=5E-5)

        # opt_dcgan = Adam(lr=1E-5)
        # opt_discriminator = Adam(lr=1E-5)

        # Load generator model
        generator_model = models.load("generator_%s" % generator,
                                      noise_dim,
                                      img_dim,
                                      bn_mode,
                                      batch_size,
                                      dset=dset)
        # Load discriminator model
        discriminator_model = models.load("DCGAN_discriminator",
                                          noise_dim,
                                          img_dim,
                                          bn_mode,
                                          batch_size,
                                          dset=dset)

        generator_model.compile(loss='mse', optimizer=opt_discriminator)
        discriminator_model.trainable = False

        DCGAN_model = models.DCGAN(generator_model,
                                   discriminator_model,
                                   noise_dim,
                                   img_dim)

        loss = [models.wasserstein]
        loss_weights = [1]
        DCGAN_model.compile(loss=loss, loss_weights=loss_weights, optimizer=opt_dcgan)

        discriminator_model.trainable = True
        discriminator_model.compile(loss=models.wasserstein, optimizer=opt_discriminator)

        gen_iterations = 0
        # Start training
        print("Start training")
        for e in range(nb_epoch):
            # Initialize progbar and batch counter
            progbar = generic_utils.Progbar(epoch_size)
            batch_counter = 1
            start = time.time()

            while batch_counter < n_batch_per_epoch:

                if gen_iterations < 25 or gen_iterations % 500 == 0:
                    disc_iterations = 100
                else:
                    disc_iterations = kwargs["disc_iterations"]

                ###################################
                # 1) Train the critic / discriminator
                ###################################
                list_disc_loss_real = []
                list_disc_loss_gen = []
                for disc_it in range(disc_iterations):

                    # Clip discriminator weights
                    for l in discriminator_model.layers:
                        weights = l.get_weights()
                        weights = [np.clip(w, -0.01, 0.01) for w in weights]
                        l.set_weights(weights)

                    X_real_batch = next(data_utils.gen_batch(X_real_train, batch_size))

                    # Create a batch to feed the discriminator model
                    X_disc_real, X_disc_gen = data_utils.get_disc_batch(X_real_batch,
                                                                        generator_model,
                                                                        batch_counter,
                                                                        batch_size,
                                                                        noise_dim,
                                                                        noise_scale=noise_scale)

                    # Update the discriminator
                    disc_loss_real = discriminator_model.train_on_batch(X_disc_real, -np.ones(X_disc_real.shape[0]))
                    disc_loss_gen = discriminator_model.train_on_batch(X_disc_gen, np.ones(X_disc_gen.shape[0]))
                    list_disc_loss_real.append(disc_loss_real)
                    list_disc_loss_gen.append(disc_loss_gen)

                    # y_disc = discriminator_model.predict(X_disc_gen, verbose=0)
                    # disc_loss = discriminator_model.train_on_batch(X_disc_real, y_disc)
                    # list_disc_loss.append(disc_loss)

                #######################
                # 2) Train the generator
                #######################
                X_gen = data_utils.get_gen_batch(batch_size, noise_dim, noise_scale=noise_scale)

                # Freeze the discriminator
                discriminator_model.trainable = False
                gen_loss = DCGAN_model.train_on_batch(X_gen, -np.ones(X_gen.shape[0]))
                # Unfreeze the discriminator
                discriminator_model.trainable = True

                gen_iterations += 1
                batch_counter += 1
                progbar.add(batch_size, values=[("Loss_D", -np.mean(list_disc_loss_real) - np.mean(list_disc_loss_gen)),
                                                ("Loss_D_real", -np.mean(list_disc_loss_real)),
                                                ("Loss_D_gen", np.mean(list_disc_loss_gen)),
                                                ("Loss_G", -gen_loss)])

                # Save images for visualization
                if batch_counter % (n_batch_per_epoch / 2) == 0:
                    data_utils.plot_generated_batch(X_real_batch, generator_model,
                                                    batch_size, noise_dim, image_dim_ordering)

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

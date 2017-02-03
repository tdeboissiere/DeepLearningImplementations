import sys
import time
import numpy as np
import models_WGAN as models
from keras.utils import generic_utils
# Utils
sys.path.append("../utils")
import general_utils
import data_utils


def train(**kwargs):
    """
    Train standard DCGAN model

    args: **kwargs (dict) keyword arguments that specify the model hyperparameters
    """

    # Roll out the parameters
    generator = kwargs["generator"]
    dset = kwargs["dset"]
    img_dim = kwargs["img_dim"]
    nb_epoch = kwargs["nb_epoch"]
    batch_size = kwargs["batch_size"]
    n_batch_per_epoch = kwargs["n_batch_per_epoch"]
    bn_mode = kwargs["bn_mode"]
    noise_dim = kwargs["noise_dim"]
    noise_scale = kwargs["noise_scale"]
    lr_D = kwargs["lr_D"]
    lr_G = kwargs["lr_G"]
    opt_D = kwargs["opt_D"]
    opt_G = kwargs["opt_G"]
    clamp_lower = kwargs["clamp_lower"]
    clamp_upper = kwargs["clamp_upper"]
    image_dim_ordering = kwargs["image_dim_ordering"]
    epoch_size = n_batch_per_epoch * batch_size

    print("\nExperiment parameters:")
    for key in kwargs.keys():
        print key, kwargs[key]
    print("\n")

    # Setup environment (logging directory etc)
    general_utils.setup_logging("DCGAN")

    # Load and normalize data
    X_real_train = data_utils.load_image_dataset(dset, img_dim, image_dim_ordering)

    # Get the full real image dimension
    img_dim = X_real_train.shape[-3:]

    # Create optimizers
    opt_G = data_utils.get_optimizer(opt_G, lr_G)
    opt_D = data_utils.get_optimizer(opt_D, lr_D)

    #######################
    # Load models
    #######################
    noise_dim = (noise_dim,)
    if generator == "upsampling":
        generator_model = models.generator_upsampling(noise_dim, img_dim, bn_mode, dset=dset)
    else:
        generator_model = models.generator_deconv(noise_dim, img_dim, bn_mode, batch_size, dset=dset)
    discriminator_model = models.discriminator(img_dim, bn_mode)
    DCGAN_model = models.DCGAN(generator_model, discriminator_model, noise_dim, img_dim)

    ############################
    # Compile models
    ############################
    generator_model.compile(loss='mse', optimizer=opt_G)
    discriminator_model.trainable = False
    DCGAN_model.compile(loss=models.wasserstein, optimizer=opt_G)
    discriminator_model.trainable = True
    discriminator_model.compile(loss=models.wasserstein, optimizer=opt_D)

    # Global iteration counter for generator updates
    gen_iterations = 0

    #################
    # Start training
    ################
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
                    weights = [np.clip(w, clamp_lower, clamp_upper) for w in weights]
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

            #######################
            # 2) Train the generator
            #######################
            X_gen = data_utils.sample_noise(noise_scale, batch_size, noise_dim)

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

            # Save images for visualization ~2 times per epoch
            if batch_counter % (n_batch_per_epoch / 2) == 0:
                data_utils.plot_generated_batch(X_real_batch, generator_model,
                                                batch_size, noise_dim, image_dim_ordering)

        print('\nEpoch %s/%s, Time: %s' % (e + 1, nb_epoch, time.time() - start))

        # Save model weights (by default, every 5 epochs)
        data_utils.save_model_weights(generator_model, discriminator_model, DCGAN_model, e)


def train_toy(**kwargs):
    """
    Train model

    args: **kwargs (dict) keyword arguments that specify the model hyperparameters
    """

    # Roll out the parameters
    batch_size = kwargs["batch_size"]
    n_batch_per_epoch = kwargs["n_batch_per_epoch"]
    nb_epoch = kwargs["nb_epoch"]
    noise_dim = kwargs["noise_dim"]
    noise_scale = kwargs["noise_scale"]
    lr_D = kwargs["lr_D"]
    lr_G = kwargs["lr_G"]
    opt_D = kwargs["opt_D"]
    opt_G = kwargs["opt_G"]
    clamp_lower = kwargs["clamp_lower"]
    clamp_upper = kwargs["clamp_upper"]
    epoch_size = n_batch_per_epoch * batch_size

    print("\nExperiment parameters:")
    for key in kwargs.keys():
        print key, kwargs[key]
    print("\n")

    # Setup environment (logging directory etc)
    general_utils.setup_logging("toy_MLP")

    # Load and rescale data
    X_real_train = data_utils.load_toy()

    # Create optimizers
    opt_G = data_utils.get_optimizer(opt_G, lr_G)
    opt_D = data_utils.get_optimizer(opt_D, lr_D)

    #######################
    # Load models
    #######################
    noise_dim = (noise_dim,)
    generator_model = models.generator_toy(noise_dim)
    discriminator_model = models.discriminator_toy()
    GAN_model = models.GAN_toy(generator_model, discriminator_model, noise_dim)

    ############################
    # Compile models
    ############################
    generator_model.compile(loss='mse', optimizer=opt_G)
    discriminator_model.trainable = False
    GAN_model.compile(loss=models.wasserstein, optimizer=opt_G)
    discriminator_model.trainable = True
    discriminator_model.compile(loss=models.wasserstein, optimizer=opt_D)

    # Global iteration counter for generator updates
    gen_iterations = 0

    #################
    # Start training
    #################
    for e in range(nb_epoch):
        # Initialize progbar and batch counter
        progbar = generic_utils.Progbar(epoch_size)
        batch_counter = 1
        start = time.time()

        while batch_counter < n_batch_per_epoch:

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
                    weights = [np.clip(w, clamp_lower, clamp_upper) for w in weights]
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

            #######################
            # 2) Train the generator
            #######################
            X_gen = data_utils.sample_noise(noise_scale, batch_size, noise_dim)

            # Freeze the discriminator
            discriminator_model.trainable = False
            gen_loss = GAN_model.train_on_batch(X_gen, -np.ones(X_gen.shape[0]))
            # Unfreeze the discriminator
            discriminator_model.trainable = True

            batch_counter += 1
            progbar.add(batch_size, values=[("Loss_D", -np.mean(list_disc_loss_real) - np.mean(list_disc_loss_gen)),
                                            ("Loss_D_real", -np.mean(list_disc_loss_real)),
                                            ("Loss_D_gen", np.mean(list_disc_loss_gen)),
                                            ("Loss_G", -gen_loss)])

            # # Save images for visualization
            if gen_iterations % 50 == 0:
                data_utils.plot_generated_toy_batch(X_real_train, generator_model,
                                                    discriminator_model, noise_dim, gen_iterations)
            gen_iterations += 1

        print('\nEpoch %s/%s, Time: %s' % (e + 1, nb_epoch, time.time() - start))

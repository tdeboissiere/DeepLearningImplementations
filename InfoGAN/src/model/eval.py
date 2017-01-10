import sys
import numpy as np
import models
import matplotlib.pylab as plt
# Utils
sys.path.append("../utils")
import data_utils
import general_utils


def eval(**kwargs):

    # Roll out the parameters
    batch_size = kwargs["batch_size"]
    generator = kwargs["generator"]
    model_name = kwargs["model_name"]
    image_dim_ordering = kwargs["image_dim_ordering"]
    img_dim = kwargs["img_dim"]
    cont_dim = (kwargs["cont_dim"],)
    cat_dim = (kwargs["cat_dim"],)
    noise_dim = (kwargs["noise_dim"],)
    bn_mode = kwargs["bn_mode"]
    noise_scale = kwargs["noise_scale"]
    dset = kwargs["dset"]
    epoch = kwargs["epoch"]

    # Setup environment (logging directory etc)
    general_utils.setup_logging(model_name)

    # Load and rescale data
    if dset == "RGZ":
        X_real_train = data_utils.load_RGZ(img_dim, image_dim_ordering)
    if dset == "mnist":
        X_real_train, _, _, _ = data_utils.load_mnist(image_dim_ordering)
    img_dim = X_real_train.shape[-3:]

    # Load generator model
    generator_model = models.load("generator_%s" % generator,
                                  cat_dim,
                                  cont_dim,
                                  noise_dim,
                                  img_dim,
                                  bn_mode,
                                  batch_size,
                                  dset=dset)

    # Load colorization model
    generator_model.load_weights("../../models/%s/gen_weights_epoch%s.h5" %
                                 (model_name, epoch))

    X_plot = []
    # Vary the categorical variable
    for i in range(cat_dim[0]):
        X_noise = data_utils.sample_noise(noise_scale, batch_size, noise_dim)
        X_cont = data_utils.sample_noise(noise_scale, batch_size, cont_dim)
        X_cont = np.repeat(X_cont[:1, :], batch_size, axis=0)  # fix continuous noise
        X_cat = np.zeros((batch_size, cat_dim[0]), dtype='float32')
        X_cat[:, i] = 1  # always the same categorical value

        X_gen = generator_model.predict([X_cat, X_cont, X_noise])
        X_gen = data_utils.inverse_normalization(X_gen)

        if image_dim_ordering == "th":
            X_gen = X_gen.transpose(0,2,3,1)

        X_gen = [X_gen[i] for i in range(len(X_gen))]
        X_plot.append(np.concatenate(X_gen, axis=1))
    X_plot = np.concatenate(X_plot, axis=0)

    plt.figure(figsize=(8,10))
    if X_plot.shape[-1] == 1:
        plt.imshow(X_plot[:, :, 0], cmap="gray")
    else:
        plt.imshow(X_plot)
    plt.xticks([])
    plt.yticks([])
    plt.ylabel("Varying categorical factor", fontsize=28, labelpad=60)

    plt.annotate('', xy=(-0.05, 0), xycoords='axes fraction', xytext=(-0.05, 1),
                 arrowprops=dict(arrowstyle="-|>", color='k', linewidth=4))
    plt.tight_layout()
    plt.savefig("../../figures/varying_categorical.png")
    plt.clf()
    plt.close()

    # Vary the continuous variables
    X_plot = []
    # First get the extent of the noise sampling
    x = np.ravel(data_utils.sample_noise(noise_scale, batch_size * 20000, cont_dim))
    # Define interpolation points
    x = np.linspace(x.min(), x.max(), num=batch_size)
    for i in range(batch_size):
        X_noise = data_utils.sample_noise(noise_scale, batch_size, noise_dim)
        X_cont = np.concatenate([np.array([x[i], x[j]]).reshape(1, -1) for j in range(batch_size)], axis=0)
        X_cat = np.zeros((batch_size, cat_dim[0]), dtype='float32')
        X_cat[:, 1] = 1  # always the same categorical value

        X_gen = generator_model.predict([X_cat, X_cont, X_noise])
        X_gen = data_utils.inverse_normalization(X_gen)
        if image_dim_ordering == "th":
            X_gen = X_gen.transpose(0,2,3,1)
        X_gen = [X_gen[i] for i in range(len(X_gen))]
        X_plot.append(np.concatenate(X_gen, axis=1))
    X_plot = np.concatenate(X_plot, axis=0)

    plt.figure(figsize=(10,10))
    if X_plot.shape[-1] == 1:
        plt.imshow(X_plot[:, :, 0], cmap="gray")
    else:
        plt.imshow(X_plot)
    plt.xticks([])
    plt.yticks([])
    plt.ylabel("Varying continuous factor 1", fontsize=28, labelpad=60)
    plt.annotate('', xy=(-0.05, 0), xycoords='axes fraction', xytext=(-0.05, 1),
                 arrowprops=dict(arrowstyle="-|>", color='k', linewidth=4))
    plt.xlabel("Varying continuous factor 2", fontsize=28, labelpad=60)
    plt.annotate('', xy=(1, -0.05), xycoords='axes fraction', xytext=(0, -0.05),
                 arrowprops=dict(arrowstyle="-|>", color='k', linewidth=4))
    plt.tight_layout()
    plt.savefig("../../figures/varying_continuous.png")
    plt.clf()
    plt.close()

import os
import h5py
import numpy as np
from scipy import stats
from keras.utils import np_utils
from keras.datasets import mnist, cifar10
import matplotlib.pylab as plt
import matplotlib.gridspec as gridspec
from keras.optimizers import Adam, SGD, RMSprop


def normalization(X, image_dim_ordering):

    X = X / 255.
    if image_dim_ordering == "tf":
        X = (X - 0.5) / 0.5
    else:
        X = (X - 0.5) / 0.5

    return X


def inverse_normalization(X):

    return ((X * 0.5 + 0.5) * 255.).astype(np.uint8)


def load_mnist(image_dim_ordering):

    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    if image_dim_ordering == 'th':
        X_train = X_train.reshape(X_train.shape[0], 1, 28, 28)
        X_test = X_test.reshape(X_test.shape[0], 1, 28, 28)
    else:
        X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
        X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)

    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')

    X_train = normalization(X_train, image_dim_ordering)
    X_test = normalization(X_test, image_dim_ordering)

    nb_classes = len(np.unique(np.hstack((y_train, y_test))))

    Y_train = np_utils.to_categorical(y_train, nb_classes)
    Y_test = np_utils.to_categorical(y_test, nb_classes)

    return X_train, Y_train, X_test, Y_test


def load_cifar10(image_dim_ordering):

    (X_train, y_train), (X_test, y_test) = cifar10.load_data()

    if image_dim_ordering == 'th':
        X_train = X_train.reshape(X_train.shape[0], 3, 32, 32)
        X_test = X_test.reshape(X_test.shape[0], 3, 32, 32)
    else:
        X_train = X_train.reshape(X_train.shape[0], 32, 32, 3)
        X_test = X_test.reshape(X_test.shape[0], 32, 32, 3)

    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')

    X_train = normalization(X_train, image_dim_ordering)
    X_test = normalization(X_test, image_dim_ordering)

    nb_classes = len(np.unique(np.vstack((y_train, y_test))))

    Y_train = np_utils.to_categorical(y_train, nb_classes)
    Y_test = np_utils.to_categorical(y_test, nb_classes)

    return X_train, Y_train, X_test, Y_test


def load_celebA(img_dim, image_dim_ordering):

    with h5py.File("../../data/processed/CelebA_%s_data.h5" % img_dim, "r") as hf:

        X_real_train = hf["data"][:].astype(np.float32)
        X_real_train = normalization(X_real_train, image_dim_ordering)

        if image_dim_ordering == "tf":
            X_real_train = X_real_train.transpose(0, 2, 3, 1)

        return X_real_train


def load_image_dataset(dset, img_dim, image_dim_ordering):

    if dset == "celebA":
        X_real_train = load_celebA(img_dim, image_dim_ordering)
    if dset == "mnist":
        X_real_train, _, _, _ = load_mnist(image_dim_ordering)
    if dset == "cifar10":
        X_real_train, _, _, _ = load_cifar10(image_dim_ordering)

    return X_real_train


def load_toy(n_mixture=8, std=0.01, radius=1.0, pts_per_mixture=5000):

    thetas = np.linspace(0, 2 * np.pi, n_mixture + 1)[:-1]
    xs, ys = radius * np.sin(thetas), radius * np.cos(thetas)
    cov = std * np.eye(2)

    X = np.zeros((n_mixture * pts_per_mixture, 2))

    for i in range(n_mixture):

        mean = np.array([xs[i], ys[i]])
        pts = np.random.multivariate_normal(mean, cov, pts_per_mixture)
        X[i * pts_per_mixture: (i + 1) * pts_per_mixture, :] = pts

    return X


def get_optimizer(opt, lr):

    if opt == "SGD":
        return SGD(lr=lr)
    elif opt == "RMSprop":
        return RMSprop(lr=lr)
    elif opt == "Adam":
        return Adam(lr=lr, beta1=0.5)


def gen_batch(X, batch_size):

    while True:
        idx = np.random.choice(X.shape[0], batch_size, replace=False)
        yield X[idx]


def sample_noise(noise_scale, batch_size, noise_dim):

    return np.random.normal(scale=noise_scale, size=(batch_size, noise_dim[0]))


def get_disc_batch(X_real_batch, generator_model, batch_counter, batch_size, noise_dim, noise_scale=0.5):

    # Pass noise to the generator
    noise_input = sample_noise(noise_scale, batch_size, noise_dim)
    # Produce an output
    X_disc_gen = generator_model.predict(noise_input)
    X_disc_real = X_real_batch[:batch_size]

    return X_disc_real, X_disc_gen


def save_model_weights(generator_model, discriminator_model, DCGAN_model, e):

    model_path = "../../models/DCGAN"

    if e % 5 == 0:
        gen_weights_path = os.path.join(model_path, '%s_epoch%s.h5' % (generator_model.name, e))
        generator_model.save_weights(gen_weights_path, overwrite=True)

        disc_weights_path = os.path.join(model_path, '%s_epoch%s.h5' % (discriminator_model.name, e))
        discriminator_model.save_weights(disc_weights_path, overwrite=True)

        DCGAN_weights_path = os.path.join(model_path, '%s_epoch%s.h5' % (DCGAN_model.name, e))
        DCGAN_model.save_weights(DCGAN_weights_path, overwrite=True)


def plot_generated_batch(X_real, generator_model, batch_size, noise_dim, image_dim_ordering, noise_scale=0.5):

    # Generate images
    X_gen = sample_noise(noise_scale, batch_size, noise_dim)
    X_gen = generator_model.predict(X_gen)

    X_real = inverse_normalization(X_real)
    X_gen = inverse_normalization(X_gen)

    Xg = X_gen[:8]
    Xr = X_real[:8]

    if image_dim_ordering == "tf":
        X = np.concatenate((Xg, Xr), axis=0)
        list_rows = []
        for i in range(int(X.shape[0] / 4)):
            Xr = np.concatenate([X[k] for k in range(4 * i, 4 * (i + 1))], axis=1)
            list_rows.append(Xr)

        Xr = np.concatenate(list_rows, axis=0)

    if image_dim_ordering == "th":
        X = np.concatenate((Xg, Xr), axis=0)
        list_rows = []
        for i in range(int(X.shape[0] / 4)):
            Xr = np.concatenate([X[k] for k in range(4 * i, 4 * (i + 1))], axis=2)
            list_rows.append(Xr)

        Xr = np.concatenate(list_rows, axis=1)
        Xr = Xr.transpose(1,2,0)

    if Xr.shape[-1] == 1:
        plt.imshow(Xr[:, :, 0], cmap="gray")
    else:
        plt.imshow(Xr)
    plt.savefig("../../figures/current_batch.png")
    plt.clf()
    plt.close()


def plot_generated_toy_batch(X_real, generator_model, discriminator_model, noise_dim, gen_iter, noise_scale=0.5):

    # Generate images
    X_gen = sample_noise(noise_scale, 10000, noise_dim)
    X_gen = generator_model.predict(X_gen)

    # Get some toy data to plot KDE of real data
    data = load_toy(pts_per_mixture=200)
    x = data[:, 0]
    y = data[:, 1]
    xmin, xmax = -1.5, 1.5
    ymin, ymax = -1.5, 1.5

    # Peform the kernel density estimate
    xx, yy = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
    positions = np.vstack([xx.ravel(), yy.ravel()])
    values = np.vstack([x, y])
    kernel = stats.gaussian_kde(values)
    f = np.reshape(kernel(positions).T, xx.shape)

    # Plot the contour
    fig = plt.figure(figsize=(10,10))
    plt.suptitle("Generator iteration %s" % gen_iter, fontweight="bold", fontsize=22)
    ax = fig.gca()
    ax.contourf(xx, yy, f, cmap='Blues', vmin=np.percentile(f,80), vmax=np.max(f), levels=np.linspace(0.25, 0.85, 30))

    # Also plot the contour of the discriminator
    delta = 0.025
    xmin, xmax = -1.5, 1.5
    ymin, ymax = -1.5, 1.5
    # Create mesh
    XX, YY = np.meshgrid(np.arange(xmin, xmax, delta), np.arange(ymin, ymax, delta))
    arr_pos = np.vstack((np.ravel(XX), np.ravel(YY))).T
    # Get Z = predictions
    ZZ = discriminator_model.predict(arr_pos)
    ZZ = ZZ.reshape(XX.shape)
    # Plot contour
    ax.contour(XX, YY, ZZ, cmap="Blues", levels=np.linspace(0.25, 0.85, 10))
    dy, dx = np.gradient(ZZ)
    # Add streamlines
    # plt.streamplot(XX, YY, dx, dy, linewidth=0.5, cmap="magma", density=1, arrowsize=1)
    # Scatter generated data
    plt.scatter(X_gen[:1000, 0], X_gen[:1000, 1], s=20, color="coral", marker="o")

    l_gen = plt.Line2D((0,1),(0,0), color='coral', marker='o', linestyle='', markersize=20)
    l_D = plt.Line2D((0,1),(0,0), color='steelblue', linewidth=3)
    l_real = plt.Rectangle((0, 0), 1, 1, fc="steelblue")

    # Create legend from custom artist/label lists
    # bbox_to_anchor = (0.4, 1)
    ax.legend([l_real, l_D, l_gen], ['Real data KDE', 'Discriminator contour',
                                     'Generated data'], fontsize=18, loc="upper left")
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax + 0.8)
    plt.savefig("../../figures/toy_dataset_iter%s.jpg" % gen_iter)
    plt.clf()
    plt.close()


if __name__ == '__main__':

    data = load_toy(pts_per_mixture=200)

    x = data[:, 0]
    y = data[:, 1]
    xmin, xmax = -1.5, 1.5
    ymin, ymax = -1.5, 1.5

    # Peform the kernel density estimate
    xx, yy = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
    positions = np.vstack([xx.ravel(), yy.ravel()])
    values = np.vstack([x, y])
    kernel = stats.gaussian_kde(values)
    f = np.reshape(kernel(positions).T, xx.shape)

    fig = plt.figure()
    gen_it = 5
    plt.suptitle("Generator iteration %s" % gen_it, fontweight="bold")
    ax = fig.gca()
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    # Contourf plot
    cfset = ax.contourf(xx, yy, f, cmap='Blues', vmin=np.percentile(f,90),
                        vmax=np.max(f), levels=np.linspace(0.25, 0.85, 30))
    # cfset = ax.contour(xx, yy, f, color="k", levels=np.linspace(0.25, 0.85, 30), label="roger")
    plt.legend()
    plt.show()

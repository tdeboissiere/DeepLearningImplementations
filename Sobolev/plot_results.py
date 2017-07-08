import matplotlib.pylab as plt
from matplotlib.pyplot import cm
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.gridspec as gridspec
import os
import numpy as np


def plot_results(args, out, out_sobolev):

    _, predict_fn, list_loss = out
    _, predict_fn_S, list_loss_S, list_loss_J_S = out_sobolev

    # Create a mesh on which to evaluate pred_fn
    X = np.arange(-5, 5, 0.25)
    Y = np.arange(-5, 5, 0.25)
    X, Y = np.meshgrid(X, Y)

    xx = X.ravel().reshape(-1, 1)
    yy = Y.ravel().reshape(-1, 1)

    inputs = np.concatenate((xx, yy), axis=1).astype(np.float32)
    Z = predict_fn(inputs).reshape(X.shape)
    Z_S = predict_fn_S(inputs).reshape(X.shape)

    fig = plt.figure(figsize=(15, 15))
    gs = gridspec.GridSpec(2,2, wspace=0.3)

    # Plot for standard network
    ax = fig.add_subplot(gs[0], projection='3d')
    ax.plot_surface(X, Y, Z, cmap="viridis",
                    linewidth=0, antialiased=False)
    ax.set_zlim(-100, 250)
    ax.zaxis.set_tick_params(pad=8)
    ax.zaxis.set_major_locator(LinearLocator(5))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    ax.set_title("Standard network (%s pts)" % args.npts, fontsize=22)

    ax = plt.subplot(gs[1])
    ax.plot(list_loss, linewidth=2, label="MSE loss")
    ax.set_xlabel("Epochs", fontsize=20)
    ax.set_ylabel("MSE loss (lower is better)", fontsize=20)
    ax.set_ylim([1, 2E3])
    ax.set_yscale("log")
    ax.set_title("Training loss standard network", fontsize=22)
    ax.legend(loc="best", fontsize=20)

    # Plot for Sobolev network
    ax = fig.add_subplot(gs[2], projection='3d')
    ax.plot_surface(X, Y, Z_S, cmap="viridis",
                    linewidth=0, antialiased=False)
    ax.set_zlim(-100, 250)
    ax.zaxis.set_tick_params(pad=8)
    ax.zaxis.set_major_locator(LinearLocator(5))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    ax.set_title("Sobolev network (%s pts)" % args.npts, fontsize=22)

    ax = plt.subplot(gs[3])
    ax.plot(list_loss_S, linewidth=2, label="MSE loss")
    ax.plot(list_loss_J_S, linewidth=2, label="Sobolev loss")
    ax.set_xlabel("Epochs", fontsize=20)
    ax.set_ylabel("MSE loss (lower is better)", fontsize=20)
    ax.set_ylim([1, 2E3])
    ax.set_yscale("log")
    ax.set_title("Training loss Sobolev network", fontsize=22)
    ax.legend(loc="best", fontsize=20)

    if not os.path.exists("figures"):
        os.makedirs("figures")

    fig_name = "plot_%s_epochs_%s_npts_%s_LR_%s_sobolev_weight.png" % (args.nb_epoch,
                                                                       args.npts,
                                                                       args.learning_rate,
                                                                       args.sobolev_weight)

    plt.savefig(os.path.join("figures", fig_name))
    plt.clf()
    plt.close()


def tang(x):

    x0, x1 = x[:, 0], x[:, 1]
    f0 = 0.5 * (np.power(x0, 4) - 16 * np.power(x0, 2) + 5 * x0)
    f1 = 0.5 * (np.power(x1, 4) - 16 * np.power(x1, 2) + 5 * x1)
    return f0 + f1


def plot_tang(X, Y, Z, title, npts=None):

    fig = plt.figure()
    ax = fig.gca(projection='3d')

    # Plot the surface.
    surf = ax.plot_surface(X, Y, Z, cmap="viridis",
                           linewidth=0, antialiased=False)

    # Customize the z axis.
    ax.set_zlim(-100, 250)
    ax.zaxis.set_tick_params(pad=8)
    ax.zaxis.set_major_locator(LinearLocator(5))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)

    if "teacher" in title:
        plt.suptitle("Teacher model")

    if "student" in title and "sobolev" not in title:
        assert (npts is not None)
        plt.suptitle("Student model %s training pts" % npts)

    if "sobolev" in title:
        assert (npts is not None)
        plt.suptitle("Student model %s training pts + Sobolev" % npts)

    else:
        plt.suptitle("Styblinski Tang function")

    plt.savefig(title)


if __name__ == '__main__':

    X = np.arange(-5, 5, 0.25)
    Y = np.arange(-5, 5, 0.25)
    X, Y = np.meshgrid(X, Y)

    xx = X.ravel().reshape(-1, 1)
    yy = Y.ravel().reshape(-1, 1)

    inpts = np.concatenate((xx, yy), axis=1)
    Z = tang(inpts).reshape(X.shape)

    plot_tang(X, Y, Z, "styblinski_tang.png")

import matplotlib.pylab as plt
import matplotlib.gridspec as gridspec
import json
import os


def plot():

    ############################
    # Adapt to existing experiments
    ############################
    list_width = [0.5, 1, 1.5, 2]
    list_depth = [6, 10, 18, 34]
    learning_rate = "1E-5"
    result_folder = "results_1e-5"

    plt.figure(figsize=(12,5))
    gs = gridspec.GridSpec(1,2)

    ##################
    # SGD results
    ##################
    ax0 = plt.subplot(gs[0])

    for i in range(len(list_depth)):
        depth = list_depth[i]
        exp = "%s/RELUNet_depth_%s_opt_SGD_drop_0_bn_True.json" % (result_folder, depth)
        with open(exp, "r") as f:
            d_losses = json.load(f)
            ax0.plot(d_losses["train_loss"],
                     linewidth=list_width[i],
                     color="C0",
                     label="RELU, Depth: %s" % depth)

    for i in range(len(list_depth)):
        depth = list_depth[i]
        exp = "%s/SELUNet_depth_%s_opt_SGD_drop_0_bn_False.json" % (result_folder, depth)
        with open(exp, "r") as f:
            d_losses = json.load(f)
            ax0.plot(d_losses["train_loss"],
                     linewidth=list_width[i],
                     color="C1",
                     label="SELU, Depth: %s" % depth)
    ax0.legend(loc="best")
    ax0.set_title("SGD, Learning Rate = %s" % learning_rate, fontsize=16)
    ax0.set_yscale("log")
    ax0.set_ylim([1E-6, 10])
    ax0.set_xlabel("Epochs", fontsize=18)
    ax0.set_ylabel("Train logloss", fontsize=18)

    ##################
    # Adam results
    ##################
    ax1 = plt.subplot(gs[1])

    for i in range(len(list_depth)):
        depth = list_depth[i]
        exp = "%s/RELUNet_depth_%s_opt_Adam_drop_0_bn_True.json" % (result_folder, depth)
        with open(exp, "r") as f:
            d_losses = json.load(f)
            ax1.plot(d_losses["train_loss"],
                     linewidth=list_width[i],
                     color="C0",
                     label="RELU, Depth: %s" % depth)

    for i in range(len(list_depth)):
        depth = list_depth[i]
        exp = "%s/SELUNet_depth_%s_opt_Adam_drop_0_bn_False.json" % (result_folder, depth)
        with open(exp, "r") as f:
            d_losses = json.load(f)
            ax1.plot(d_losses["train_loss"],
                     linewidth=list_width[i],
                     color="C1",
                     label="SELU, Depth: %s" % depth)
    ax1.legend(loc="best")
    ax1.set_title("Adam, Learning Rate = %s" % learning_rate, fontsize=16)
    ax1.set_yscale("log")
    ax1.set_ylim([1E-6, 10])
    ax1.set_xlabel("Epochs", fontsize=18)
    ax1.set_ylabel("Train logloss", fontsize=18)

    if not os.path.exists("figures"):
        os.makedirs("figures")

    plt.savefig("figures/SELU_LR_%s.png" % learning_rate)

if __name__ == '__main__':

    plot()

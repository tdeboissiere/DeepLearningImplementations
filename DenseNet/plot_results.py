import matplotlib.pylab as plt
import json
import numpy as np


def plot_cifar10():

    with open("experiment_log_cifar10.json", "r") as f:
        d = json.load(f)

    train_accuracy = 100 * (1 - np.array(d["train_loss"])[:, 1])
    test_accuracy = 100 * (1 - np.array(d["test_loss"])[:, 1])

    plt.plot(train_accuracy, color="tomato", linewidth=2)
    plt.plot(test_accuracy, color="steelblue", linewidth=2)

    plt.grid()
    plt.show()
    plt.clf()
    plt.close()

if __name__ == '__main__':
    plot_cifar10()

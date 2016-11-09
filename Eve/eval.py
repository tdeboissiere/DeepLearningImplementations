import json
import glob
import argparse
import matplotlib.pylab as plt


def plot_results(list_log, to_plot="losses"):

    list_color = [u'#E24A33',
                  u'#348ABD',
                  u'#FBC15E',
                  u'#777777',
                  u'#988ED5',
                  u'#8EBA42',
                  u'#FFB5B8']

    plt.figure()
    for idx, log in enumerate(list_log):
        with open(log, "r") as f:
            d = json.load(f)

            experiment_name = d["experiment_name"]
            color = list_color[idx]

            plt.plot(d["train_%s" % to_plot],
                     color=color,
                     linewidth=3,
                     label="Train %s" % experiment_name)
            plt.plot(d["val_%s" % to_plot],
                     color=color,
                     linestyle="--",
                     linewidth=3,)
    plt.ylabel(to_plot, fontsize=20)
    if to_plot == "losses":
        plt.yscale("log")
    if to_plot == "accs":
        plt.ylim([0, 1.1])
    plt.xlabel("Number of epochs", fontsize=20)
    plt.title("%s experiment" % dataset, fontsize=22)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig("./figures/%s_results_%s.png" % (dataset, to_plot))
    plt.show()


if __name__ == '__main__':

    list_log = glob.glob("./log/*.json")

    parser = argparse.ArgumentParser(description='Plot results of experiments')
    parser.add_argument('dataset', type=str,
                        help='name of the dataset: cifar10, cifar100 or mnist')
    parser.add_argument('--to_plot', type=str, default="losses",
                        help='metric to plot: losses (log loss) or accs (accuracies)')
    args = parser.parse_args()
    dataset = args.dataset

    list_log = [l for l in list_log if dataset in l]

    plot_results(list_log, to_plot=args.to_plot)

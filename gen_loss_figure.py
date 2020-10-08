import os
import argparse
import matplotlib.pyplot as plt
import numpy as np

def plot_loss(experiment):
    log_file_path = "models/{}/stackexchange.log".format(experiment)
    steps_list, acc_list = [], []

    with open(log_file_path, "r") as file:
        for line in file:
            line_split = line.split()
            if ("Step" in line_split):
                step_num = line_split[4].split('/')[0]
                acc = line_split[6][:-1]
                if (int(step_num) not in steps_list):
                    steps_list.append(int(step_num))
                    acc_list.append(float(acc))

    savepath = "output/{}/{}-loss.png".format(experiment, experiment)
    plt.plot(steps_list, acc_list)
    plt.title(experiment)
    plt.xlabel("Train steps")
    plt.ylabel("Accuracy")
    plt.yticks(np.arange(0, 100, 10))
    plt.savefig(savepath)


def main(config):
    experiment = config.experiment.lower()
    plot_loss(experiment)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment', type=str, default="stackexchange_0")
    config = parser.parse_args()

    main(config)

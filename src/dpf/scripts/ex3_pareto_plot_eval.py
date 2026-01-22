import pickle

import numpy as np
from matplotlib import pyplot as plt


def main():
    number_of_seeds = 10
    # optimizers = ["Adam", "RMSprop"]
    optimizers = ["Adam"]

    dc_times = []
    nr_times = []
    dc_losses = []
    nr_losses = []
    dc_voltage = []
    nr_voltage = []
    x_min = 0
    y_min = 10**1000
    x_max = -(10**1000)
    y_max = -(10**1000)

    for seed in range(number_of_seeds):
        with open(f"out/temp/ex3_DC_{seed}.pkl", "rb") as readFile:
            results = pickle.load(readFile)
            for x in results["times"]:
                dc_times.append(x)
            for x in results["mismatches"]:
                dc_losses.append(x)

            x_max = max(x_max, max(dc_times))
            y_min = min(y_min, min(dc_losses))
            y_max = max(y_max, max(dc_losses))

        # dc_voltage = results["voltage"]
        # print(results)
    for seed in range(number_of_seeds):
        with open(f"out/temp/ex3_NR_{seed}.pkl", "rb") as readFile:
            results = pickle.load(readFile)
            for x in results["times"]:
                nr_times.append(x)
            for x in results["mismatches"]:
                nr_losses.append(x)

            x_max = max(x_max, max(nr_times))
            y_min = min(y_min, min(nr_losses))
            y_max = max(y_max, max(nr_losses))
            # nr_voltage = results["voltage"]

    dc_times = np.array(dc_times)
    nr_times = np.array(nr_times)
    dc_losses = np.array(dc_losses)
    nr_losses = np.array(nr_losses)

    avg_dc_times = dc_times.mean(axis=0)
    avg_nr_times = nr_times.mean(axis=0)
    avg_dc_losses = dc_losses.mean(axis=0)
    avg_nr_losses = nr_losses.mean(axis=0)

    plt.figure()
    # plt.axhline(0)
    # plt.axvline(0)
    # plt.yscale("log")
    plt.plot(avg_dc_times, avg_dc_losses, label="DC", color="blue", marker="o")
    plt.plot(avg_nr_times, avg_nr_losses, label="NR", color="red", marker="s")

    colors = ["green", "purple", "brown"]

    # create plots
    for i, optimizer in enumerate(optimizers):
        to_times = []
        to_times_without_opt_init = None
        to_losses = []
        to_voltage = None

        for seed in range(number_of_seeds):
            with open(f"out/temp/ex3_TO_{optimizer}_{seed}.pkl", "rb") as readFile:
                results = pickle.load(readFile)

                to_times.append(results["times"][0:200])
                to_losses.append(results["mismatches"][0:200])

                # to_voltage = results["voltage"]
                # to_times_without_opt_init = results["times_without_opt_init"]

        to_times = np.array(to_times)
        to_losses = np.array(to_losses)

        avg_to_times = to_times.mean(axis=0)
        avg_to_losses = to_losses.mean(axis=0)
        std_to_times = to_times.std(axis=0)
        std_to_losses = to_losses.std(axis=0)

        x_max = max(x_max, max(avg_to_times))
        y_min = min(y_min, min(avg_to_losses))
        y_max = max(y_max, max(avg_to_losses))

        # only 100 iterations
        plt.plot(
            avg_to_times,
            avg_to_losses,
            label=optimizer,
            color=colors[i],
            marker="^",
            markersize=4,
        )

    plt.xlim([0, x_max])
    # plt.ylim([y_min, y_max])

    plt.xlabel("Average Time (s)")
    plt.ylabel("Average MSE Loss")
    plt.title(f"Pareto-plot for IEEE-118 bus system averaged over 10 grid states")
    plt.legend()
    plt.savefig(f"out/plots/ex3_pareto_plot.png")
    plt.close()


if __name__ == "__main__":
    main()

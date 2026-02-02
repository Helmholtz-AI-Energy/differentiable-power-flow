"""
In ex4_scaling.py the loading time is the main factor. Here, we want to find a use case where the loading is
not the limiting factor, e.g. look at
a) Ignore the loading time for now
b) Time Series
c) Cascading Failure Analysis
"""

import pickle
import matplotlib.pyplot as plt


def main():
    plt.rcParams.update(
        {
            "font.size": 14,
            "axes.labelsize": 16,
            "axes.titlesize": 16,
            "legend.fontsize": 14,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
            "lines.linewidth": 2,
            "lines.markersize": 6,
        }
    )

    # run this to only look at the largest grid with the accuracies.
    sample = 2
    use_gpu = False
    case_names = [
        "case118",
        "case_illinois200",
        "case300",
        "case1354pegase",
        "case1888rte",
        "case2869pegase",
        "case3120sp",
        "case6495rte",
        "case6515rte",
        "case9241pegase",
    ]
    grid_sizes = [118, 200, 300, 1354, 1888, 2869, 3120, 6495, 6515, 9241]

    dc_times = []
    nr_times = []
    dc_losses = []
    nr_losses = []
    optimizers = ["Adam"]

    plt.figure()
    # case_name = case_names[-1]
    case_name = "case9241pegase"  # "case118" "case9241pegase"
    with open(f"out/temp/ex4b_DC_{case_name}_{use_gpu}.pkl", "rb") as readFile:
        results = pickle.load(readFile)
        for x in results["times"]:
            dc_times.append(x)
        for x in results["mismatches"]:
            dc_losses.append(x)

    with open(f"out/temp/ex4b_NR_{case_name}_{use_gpu}.pkl", "rb") as readFile:
        results = pickle.load(readFile)
        for x in results["times"]:
            nr_times.append(x)
        for x in results["mismatches"]:
            nr_losses.append(x)
    plt.plot(dc_times, dc_losses, label="DC", color="blue", marker="o")
    plt.plot(nr_times, nr_losses, label="NR", color="red", marker="s")

    to_times = []
    to_losses = []

    with open(
        f"out/temp/ex4b_TO_{optimizers[0]}_{case_name}_{use_gpu}.pkl", "rb"
    ) as readFile:
        results = pickle.load(readFile)
        to_times.append(results["times"])
        to_losses.append(results["mismatches"])

    iterations_to_plot = 100
    plt.plot(
        to_times[-1][:iterations_to_plot],
        to_losses[-1][:iterations_to_plot],
        label="DPF",
        color="green",
        marker="s",
        markersize=3,
    )

    plt.xlabel("Time [s]")
    plt.ylabel("MSE loss")
    plt.title(f"Pareto-plot of Power-flow methods on {case_name}")
    plt.legend()
    plt.savefig(f"out/plots/ex4_largest_grid_{use_gpu}.png")
    plt.close()

    optimizers = ["Adam"]

    dc_times = []
    nr_times = []
    dc_losses = []
    nr_losses = []

    plt.figure()

    for case_name in case_names:
        with open(f"out/temp/ex4b_DC_{case_name}_{use_gpu}.pkl", "rb") as readFile:
            results = pickle.load(readFile)
            for x in results["times"]:
                dc_times.append(x)
            for x in results["mismatches"]:
                dc_losses.append(x)
    for case_name in case_names:
        with open(f"out/temp/ex4b_NR_{case_name}_{use_gpu}.pkl", "rb") as readFile:
            results = pickle.load(readFile)
            for x in results["times"]:
                nr_times.append(x)
            for x in results["mismatches"]:
                nr_losses.append(x)

    plt.plot(grid_sizes, dc_times, label="DC", color="blue", marker="o", markersize=3)
    plt.plot(grid_sizes, nr_times, label="NR", color="red", marker="s", markersize=3)
    print("dc time in ms for IEEE118:", dc_times[0] * 1000.0)
    print("nr time in ms for IEEE118:", nr_times[0] * 1000.0)

    colors = ["green", "purple", "brown"]

    # create plots
    for i, optimizer in enumerate(optimizers):
        to_times = []
        to_losses = []

        # create a subplot where only iteration many iterations are done with TO
        for case_name in case_names:
            with open(
                f"out/temp/ex4b_TO_{optimizer}_{case_name}_{use_gpu}.pkl", "rb"
            ) as readFile:
                results = pickle.load(readFile)
                to_times.append(results["times"])
                to_losses.append(results["mismatches"])

        iterations_to_report = [0, 5, 10, 15, 20, 25, 30]
        print(
            "time per iteration in ms for IEEE118:",
            (to_times[0][20] - to_times[0][10]) / 10.0 * 1000,
        )
        for j, iteration in enumerate(
            iterations_to_report
        ):  # list of iterations to report
            if j == 0:
                plt.plot(
                    grid_sizes,
                    [x[iteration] for x in to_times],
                    label="DPF (#iterations)",
                    color=colors[i],
                    marker="^",
                    markersize=3,
                )
            plt.plot(
                grid_sizes,
                [x[iteration] for x in to_times],
                color=colors[i],
                marker="^",
                markersize=3,
            )

        for j, iteration in enumerate(iterations_to_report):
            plt.annotate(
                "(" + str(iteration) + ")",
                (grid_sizes[-1], to_times[-1][iteration]),
                textcoords="offset points",
                xytext=(-1, 2),
                fontsize=12,
            )

    plt.xlabel("Grid Size [#buses]")
    plt.ylabel("Average Time over 10 seeds in s")
    plt.title(f"Scalability")
    plt.legend()
    plt.savefig(f"out/plots/ex4b_scalability_{use_gpu}.png")
    plt.close()


if __name__ == "__main__":
    main()

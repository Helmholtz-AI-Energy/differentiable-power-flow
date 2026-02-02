import pickle

from matplotlib import pyplot as plt


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

    max_iter_nr = (
        6  # for case9241pegase this is enough, for random inputs more might be needed
    )
    max_iter_to = 1000

    # strategy = "no_connections"
    # strategy_amount_param = 0

    strategy = "total_random"  # random connections anywhere
    strategy_amount_param = 20

    use_gpu = False
    device = "gpu" if use_gpu else "cpu"

    batch_sizes_to_report = [1, 2, 4, 8, 16, 20, 32, 40, 60, 64, 80, 100, 120, 128]
    batch_sizes_to_report = [1, 2, 4, 8, 16, 32, 64, 128, 200, 256]

    nr_times = []
    to_times = []

    for batch_size in batch_sizes_to_report:
        with open(
            f"out/temp/ex9a_{batch_size}_{strategy}_{strategy_amount_param}.pkl", "rb"
        ) as readFile:
            results = pickle.load(readFile)
            nr_times.append(results["times"])

        with open(
            f"out/temp/ex9b_{use_gpu}_{max_iter_to}_{batch_size}_{strategy}_{strategy_amount_param}.pkl",
            "rb",
        ) as readFile:
            results = pickle.load(readFile)
            to_times.append(results["times"])

    plt.plot(
        batch_sizes_to_report,
        nr_times,
        label="NR",
        color="red",
        marker="s",
        markersize=3,
    )

    iterations_to_report = [250, 500, 750, 1000]
    for j, iteration in enumerate(iterations_to_report):  # list of iterations to report
        if j == 0:
            plt.plot(
                batch_sizes_to_report,
                [x[iteration - 1] for x in to_times],
                label="DPF (#iterations)",
                color="green",
                marker="^",
                markersize=3,
            )
        plt.plot(
            batch_sizes_to_report,
            [x[iteration - 1] for x in to_times],
            color="green",
            marker="^",
            markersize=3,
        )

    for j, iteration in enumerate(iterations_to_report):
        plt.annotate(
            "(" + str(iteration) + ")",
            (batch_sizes_to_report[-1], to_times[-1][iteration - 1]),
            textcoords="offset points",
            xytext=(-1, 2),
            fontsize=12,
        )

    plt.xlabel("Grid Size in multiples of 9241")
    plt.ylabel("Time in s")
    plt.title(f"Node scaling comparison")
    plt.legend()
    # plt.show()
    plt.savefig(f"out/plots/ex9c_supergrid_{strategy}_{strategy_amount_param}.png")
    plt.close()


if __name__ == "__main__":
    main()

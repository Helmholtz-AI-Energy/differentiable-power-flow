import pickle

from matplotlib import pyplot as plt


def main():
    batch_sizes_to_report = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
    use_gpu = False
    device = "gpu" if use_gpu else "cpu"
    max_iter = 30

    nr_times = []
    to_times = []

    for batch_size in batch_sizes_to_report:
        with open(f"out/temp/ex6a_{batch_size}.pkl", "rb") as readFile:
            results = pickle.load(readFile)
            nr_times.append(results["times"])
        with open(
            f"out/temp/ex6b_{use_gpu}_{max_iter}_{batch_size}.pkl", "rb"
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

    iterations_to_report = [0, 5, 10, 15, 20, 25, 30]

    for j, iteration in enumerate(iterations_to_report):  # list of iterations to report
        if j == 0:
            plt.plot(
                batch_sizes_to_report,
                [x[iteration] for x in to_times],
                label="DPF (#iterations)",
                color="green",
                marker="^",
                markersize=3,
            )
        plt.plot(
            batch_sizes_to_report,
            [x[iteration] for x in to_times],
            color="green",
            marker="^",
            markersize=3,
        )

    for j, iteration in enumerate(iterations_to_report):
        plt.annotate(
            "(" + str(iteration) + ")",
            (batch_sizes_to_report[-1], to_times[-1][iteration]),
            textcoords="offset points",
            xytext=(-1, 2),
            fontsize=9,
        )

    plt.xlabel("Grid Size in multiples of 9241")
    plt.ylabel("Time (init excluded) in s")
    plt.title(f"Scalability of NR vs DPF")
    plt.legend()
    # plt.show()
    plt.savefig(f"out/plots/ex6c_scalability.png")
    plt.close()


if __name__ == "__main__":
    main()

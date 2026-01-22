import pickle
import matplotlib.pyplot as plt


def main():
    batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
    inferred_grid_sizes = [batch_size * 9241 for batch_size in batch_sizes]
    print(inferred_grid_sizes)

    use_gpu = False

    batch_sizes_to_report = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
    max_iter = 30

    device = "gpu" if use_gpu else "cpu"

    to_times = []
    for batch_size in batch_sizes_to_report:
        with open(
            f"out/temp/ex6b_{use_gpu}_{max_iter}_{batch_size}.pkl", "rb"
        ) as readFile:
            results = pickle.load(readFile)
            to_times.append(results["times"])

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
    plt.ylabel("Time in s")
    plt.title(f"Scalability on {device}")
    plt.legend()
    # plt.show()
    plt.savefig(f"out/plots/ex6b_scalability_{use_gpu}.png")
    plt.close()


if __name__ == "__main__":
    main()

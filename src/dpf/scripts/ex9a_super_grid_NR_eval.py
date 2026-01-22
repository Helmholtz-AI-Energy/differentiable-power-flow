import pickle
import matplotlib.pyplot as plt


def main():
    print("Starting experiment 9a")

    strategy = "total_random"  # random connections anywhere
    strategy_amount_param = (
        20  # nb new connections = strategy_amount_param * batch_size
    )

    batch_sizes_to_report = [1, 2, 4, 8, 16, 32, 64, 128, 200, 256]

    nr_times = []
    for batch_size in batch_sizes_to_report:
        with open(
            f"out/temp/ex9a_{batch_size}_{strategy}_{strategy_amount_param}.pkl", "rb"
        ) as readFile:
            results = pickle.load(readFile)
            nr_times.append(results["times"])

    plt.plot(
        batch_sizes_to_report,
        nr_times,
        label="NR",
        color="red",
        marker="s",
        markersize=3,
    )

    plt.xlabel("Grid Size in multiples of 9241")
    plt.ylabel("Time in s")
    plt.title(f"Supergrid run-time using NR")
    plt.legend()
    # plt.show()
    plt.savefig(f"out/plots/ex9a_supergrid_{strategy}_{strategy_amount_param}.png")
    plt.close()


if __name__ == "__main__":
    main()

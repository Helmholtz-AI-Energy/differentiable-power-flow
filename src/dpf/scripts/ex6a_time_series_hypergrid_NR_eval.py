import pickle
import matplotlib.pyplot as plt


def main():
    print("Starting experiment 6a")

    ybus_scaling_method = "block_diagonal"  # "block_diagonal" "random"
    density = 0.5  # this parameter is ignored for block_diagonal

    batch_sizes_to_report = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]

    nr_times = []
    for batch_size in batch_sizes_to_report:
        with open(f"out/temp/ex6a_{batch_size}_{ybus_scaling_method}_{density}.pkl",
                  "rb") as readFile:
            results = pickle.load(readFile)
            nr_times.append(results["times"])

    plt.plot(batch_sizes_to_report, nr_times, label="NR", color="red", marker="s", markersize=3)

    plt.xlabel("Grid Size in multiples of 9241")
    plt.ylabel("Time (init excluded) in s")
    plt.title(f"Scalability of NR on cpu")
    plt.legend()
    # plt.show()
    plt.savefig(f"out/plots/ex6a_scalability.png")
    plt.close()


if __name__ == "__main__":
    main()

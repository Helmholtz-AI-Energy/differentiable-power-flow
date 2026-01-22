import pickle
import matplotlib.pyplot as plt


def main():
    print("Starting experiment 7b")

    max_iter = 1000  # for case9241pegase this is enough, for random inputs more might be needed

    nbs_extra_connections_to_report = [20000, 20000, 40000, 60000, 80000, 100000]

    to_times = []
    for nb_extra_connections in nbs_extra_connections_to_report:
        with open(
            f"out/temp/ex7b_{nb_extra_connections}_{max_iter}.pkl", "rb"
        ) as readFile:
            results = pickle.load(readFile)
            to_times.append(results["times"])

    plt.plot(
        nbs_extra_connections_to_report,
        to_times,
        label="DPF",
        color="red",
        marker="s",
        markersize=3,
    )

    plt.xlabel(" Number of added connections")
    plt.ylabel("Time (init excluded) in s")
    plt.title(f"Time with extra connections")
    plt.legend()
    # plt.show()
    plt.savefig(f"out/plots/ex7b_scalability.png")
    plt.close()


if __name__ == "__main__":
    main()

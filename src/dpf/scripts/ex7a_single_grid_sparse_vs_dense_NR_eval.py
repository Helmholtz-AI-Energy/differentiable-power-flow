import pickle
import matplotlib.pyplot as plt


def main():
    nbs_extra_connections_to_report = [
        500,
        1000,
        1500,
        2000,
        2500,
        3000,
        3500,
        4000,
        4500,
        5000,
    ]
    nr_times = []
    for nb_extra_connections in nbs_extra_connections_to_report:
        with open(f"out/temp/ex7a_{nb_extra_connections}.pkl", "rb") as readFile:
            results = pickle.load(readFile)
            nr_times.append(results["times"])

    plt.plot(
        nbs_extra_connections_to_report,
        nr_times,
        label="NR",
        color="red",
        marker="s",
        markersize=3,
    )

    plt.xlabel(" Number of added connections")
    plt.ylabel("Time (init excluded) in s")
    plt.title(f"Time with extra connections")
    plt.legend()
    # plt.show()
    plt.savefig(f"out/plots/ex7a_scalability.png")
    plt.close()


if __name__ == "__main__":
    main()

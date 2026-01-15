import pickle
import matplotlib.pyplot as plt

def main():
    nbs_extra_connections_to_report = [500, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000]

    to_times = []
    nr_times = []

    for nb_extra_connections in nbs_extra_connections_to_report:

        with open(f"out/temp/ex7a_{nb_extra_connections}.pkl",
                  "rb") as readFile_nr:
            results_nr = pickle.load(readFile_nr)
            nr_times.append(results_nr["times"])

        with open(f"out/temp/ex7b_{nb_extra_connections}_{1000}.pkl",
                  "rb") as readFile_to:
            results_to = pickle.load(readFile_to)
            to_times.append(results_to["times"])

    print(to_times)
    plt.plot(nbs_extra_connections_to_report, nr_times, label="NR with 7 iterations", color="red", marker="s", markersize=3)
    plt.plot(nbs_extra_connections_to_report, to_times, label="DPF with 1000 iterations", color="green", marker="s", markersize=3)

    plt.xlabel(" Number of added connections")
    plt.ylabel("Time (init excluded) in s")
    plt.title(f"Power-flow time for case9241pegase with additional random connections")
    plt.legend()
    # plt.show()
    plt.savefig(f"out/plots/ex7c_scalability.png")

    plt.close()

if __name__ == "__main__":
    main()

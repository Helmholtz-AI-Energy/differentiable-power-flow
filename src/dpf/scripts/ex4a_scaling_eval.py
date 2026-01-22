import pickle
import matplotlib.pyplot as plt


def create_times_plot():
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
    optimizers = ["Adam"]

    dc_times = []
    nr_times = []
    dc_losses = []
    nr_losses = []

    plt.figure()

    for case_name in case_names:
        with open(f"out/temp/ex4_DC_{case_name}.pkl", "rb") as readFile:
            results = pickle.load(readFile)
            for x in results["times"]:
                dc_times.append(x)
            for x in results["mismatches"]:
                dc_losses.append(x)
    for case_name in case_names:
        with open(f"out/temp/ex4_NR_{case_name}.pkl", "rb") as readFile:
            results = pickle.load(readFile)
            for x in results["times"]:
                nr_times.append(x)
            for x in results["mismatches"]:
                nr_losses.append(x)

    plt.plot(grid_sizes, dc_times, label="DC", color="blue", marker="o")
    plt.plot(grid_sizes, nr_times, label="NR", color="red", marker="s")

    colors = ["green", "purple", "brown"]

    # create plots
    for i, optimizer in enumerate(optimizers):
        to_times = []
        to_losses = []

        # create a subplot where only iteration many iterations are done with TO
        for case_name in case_names:
            with open(f"out/temp/ex4_TO_{optimizer}_{case_name}.pkl", "rb") as readFile:
                results = pickle.load(readFile)
                to_times.append(results["times"])
                to_losses.append(results["mismatches"])

        for iteration in [0]:
            plt.plot(
                grid_sizes,
                [x[iteration] for x in to_times],
                label="DPF",
                color=colors[i],
                marker="^",
            )

        print(
            [x[109] - x[9] for x in to_times]
        )  # loading the grid takes most time, the iterations are fast

    plt.xlabel("Grid Size")
    plt.ylabel("Time with grid loading [s]")
    plt.title(f"Scalability")
    plt.legend()
    plt.savefig(f"out/plots/ex4_scalability.png")
    plt.close()


def create_large_grid_plot():
    # plot for large grid only

    to_times = []
    to_losses = []
    to_normalized_losses = []

    nr_times = []
    nr_times = []
    nr_normalized_losses = []

    dc_times = []
    dc_losses = []
    dc_normalized_losses = []

    load_time = None  # only look at differences here

    with open(f"out/temp/ex4_TO_Adam_case9241pegase.pkl", "rb") as readFile:
        results = pickle.load(readFile)
        to_times.append(results["times"])
        to_losses.append(results["mismatches"])
        first_time_step = to_times[0][0]
    with open(f"out/temp/ex4_NR_case9241pegase.pkl", "rb") as readFile:
        results = pickle.load(readFile)
        for x in results["times"]:
            nr_times.append(x)
        for x in results["avg_percentage_diffes"]:
            nr_normalized_losses.append(x)
    with open(f"out/temp/ex4_DC_case9241pegase.pkl", "rb") as readFile:
        results = pickle.load(readFile)
        for x in results["times"]:
            dc_times.append(x)
        for x in results["mismatches"]:
            dc_losses.append(x)
        for x in results["avg_percentage_diffes"]:
            dc_normalized_losses.append(x)
    with open(f"out/temp/ex4_TO_Adam_case9241pegase_metrics.pkl", "rb") as readFile:
        results = pickle.load(readFile)
        to_normalized_losses.append(results["avg_percentage_diff"])

    to_times[0] = [x - first_time_step for x in to_times[0]]
    nr_times[0] = nr_times[0] - first_time_step
    dc_times[0] = dc_times[0] - first_time_step

    plt.plot(
        to_times[0][:-1],
        100 * to_normalized_losses[0],
        label="DPF",
        color="green",
        marker="^",
        linewidth=1,
        markersize=1,
    )
    plt.plot(
        dc_times[0], 100 * dc_normalized_losses[0], label="DC", color="blue", marker="o"
    )
    plt.plot(
        nr_times[0], 100 * nr_normalized_losses[0], label="NR", color="red", marker="s"
    )
    plt.xlabel("Time difference from first iteration in s")
    plt.ylabel("Average flow difference in %")
    plt.title(f"Pareto plot of power-flow methods on case9241pegase")

    plt.savefig(f"out/plots/ex4_large_grid.png")
    plt.close()


def create_small_grid_plot():
    # plot for grid 118

    to_times = []
    to_losses = []
    to_normalized_losses = []

    nr_times = []
    nr_times = []
    nr_normalized_losses = []

    dc_times = []
    dc_losses = []
    dc_normalized_losses = []

    load_time = None  # only look at differences here

    with open(f"out/temp/ex4_TO_Adam_case118.pkl", "rb") as readFile:
        results = pickle.load(readFile)
        to_times.append(results["times"])
        to_losses.append(results["mismatches"])
        first_time_step = to_times[0][0]
    with open(f"out/temp/ex4_NR_case118.pkl", "rb") as readFile:
        results = pickle.load(readFile)
        for x in results["times"]:
            nr_times.append(x)
        for x in results["avg_percentage_diffes"]:
            nr_normalized_losses.append(x)
    with open(f"out/temp/ex4_DC_case118.pkl", "rb") as readFile:
        results = pickle.load(readFile)
        for x in results["times"]:
            dc_times.append(x)
        for x in results["mismatches"]:
            dc_losses.append(x)
        for x in results["avg_percentage_diffes"]:
            dc_normalized_losses.append(x)
    with open(f"out/temp/ex4_TO_Adam_case118_metrics.pkl", "rb") as readFile:
        results = pickle.load(readFile)
        to_normalized_losses.append(results["avg_percentage_diff"])

    to_times[0] = [x - first_time_step for x in to_times[0]]
    nr_times[0] = nr_times[0] - first_time_step
    dc_times[0] = dc_times[0] - first_time_step

    plt.plot(
        to_times[0][:-1],
        100 * to_normalized_losses[0],
        label="DPF",
        color="green",
        marker="^",
        linewidth=1,
        markersize=1,
    )
    plt.plot(
        dc_times[0], 100 * dc_normalized_losses[0], label="DC", color="blue", marker="o"
    )
    plt.plot(
        nr_times[0], 100 * nr_normalized_losses[0], label="NR", color="red", marker="s"
    )
    plt.xlabel("Time difference from first iteration in s")
    plt.ylabel("Average flow difference in %")
    plt.title(f"Pareto plot of power-flow methods on case118")

    plt.savefig(f"out/plots/ex4_small_grid.png")
    plt.close()


def main():
    create_times_plot()
    create_large_grid_plot()
    create_small_grid_plot()


if __name__ == "__main__":
    main()

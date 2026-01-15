
import pickle
import matplotlib.pyplot as plt

import numpy as np


def main():

    use_gpu = False

    case_names = ["case118", "case_illinois200", "case300", "case1354pegase", "case1888rte",
                  "case2869pegase", "case3120sp", "case6495rte", "case6515rte", "case9241pegase"]
    grid_sizes = [118, 200, 300, 1354, 1888, 2869, 3120, 6495, 6515, 9241]

    to_optimizer = "Adam"


    nr_times = []
    nr_normalized_losses = []
    nr_losses = []
    to_times = []
    to_normalized_losses = []
    to_losses = []

    plt.figure()

    for case_name in case_names:

        with open(f"out/temp/ex10a_TO_{to_optimizer}_{case_name}_{use_gpu}.pkl", "rb") as readFile:
            results = pickle.load(readFile)
            to_times.append(results["times"])
            to_normalized_losses.append(results["avg_percentage_diff"])
            to_losses.append(results["loss"])

    running_min_to_normalized_losses = np.minimum.accumulate(to_normalized_losses, axis=1)  # use best loss
    running_min_to_losses = np.minimum.accumulate(to_losses, axis=1)

    iterations_to_plot = [1, 1000, 2000, 4000, 8000]
    for iter in iterations_to_plot:
        # x: grid size
        # y: losses at iteration iter and grid size x
        normalized_losses_at_iter = [100 * x[iter - 1] for x in running_min_to_normalized_losses]
        losses_at_iter = [100 * x[iter - 1] for x in running_min_to_losses]

        plt.plot(grid_sizes, losses_at_iter, label=f"Iter {iter}")
        plt.yscale("log")
        plt.legend()

    plt.xlabel("Grid Size")
    plt.ylabel("Total MSE loss")
    plt.title("Effect of grid size on convergence behaviour")
    plt.savefig(f"out/plots/ex10a.svg", format="svg")
    plt.savefig(f"out/plots/ex10a.png", format="png")
    plt.close()

    #######
    # make figure of "when loss reached"
    thresholds_to_reach = [0.1 * 2**(5-i) for i in range(0, 5)]
    # print(thresholds_to_reach)

    to_losses = np.array(to_losses)
    first_founds = []
    for threshold in thresholds_to_reach:
        # to_losses has shape [10,1000]
        first_found_for_this_threshold = []
        for to_loss_list in to_losses:
            # to_loss_list has shape [1000]
            indices_where_smaller = np.argwhere(to_loss_list < threshold)
            if len(indices_where_smaller) > 0:
                first_found = np.min(indices_where_smaller)
            else:
                first_found = None
            first_found_for_this_threshold.append(first_found)
        #print(first_found_for_this_threshold)
        first_founds.append(first_found_for_this_threshold)
    first_founds = np.array(first_founds)

    # first founds has shape (threshold, cases)

    # plot
    # print(thresholds_to_reach)
    for (i, threshold) in enumerate(thresholds_to_reach):
        plt.plot(grid_sizes, first_founds[i, :], label=f"threshold: {threshold}")
        plt.xlabel("grid size")
        plt.ylabel("iterations needed")

    plt.legend()
    plt.savefig(f"out/plots/ex10a_iterations_needed.svg", format="svg")
    plt.savefig(f"out/plots/ex10a_iterations_needed.png", format="png")
    plt.close()




    plt.figure()

    for case_name in case_names:
        with open(f"out/temp/ex10a_NR_{case_name}_{use_gpu}.pkl", "rb") as readFile:
            results = pickle.load(readFile)
            nr_times.append(results["times"])
            nr_normalized_losses.append(results["avg_percentage_diff"])
            nr_losses.append(results["losses"])

    # x: grid size
    # y: loss at iteration iter and grid size x

    plt.plot(grid_sizes, nr_normalized_losses)
    plt.legend()

    plt.xlabel("Grid Size")
    plt.ylabel("Mean flow deviation in %")
    plt.title("NR losses")
    plt.savefig(f"out/plots/ex10a_NR.svg", format="svg")
    plt.savefig(f"out/plots/ex10a_NR.png", format="png")
    plt.close()




if __name__ == "__main__":
    main()

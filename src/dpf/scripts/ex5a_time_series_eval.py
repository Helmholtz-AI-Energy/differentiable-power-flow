import pickle
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker


def main():
    start_iter = 1000
    max_iter = 300

    with open(f"out/temp/ex5_time_series.pkl", "rb") as readTOFile:
        results = pickle.load(readTOFile)
        losses = results["losses"]
        avg_diffes = results["average_percentage_diffes"]

    ######### first plot #########

    # show initial losses with new injections

    # start_iter = max_iter = 1000 for this

    start_step = 0  # fixed
    num_steps_shown = 10  # fixed
    step_size = 1
    x_vals = [int(i) for i in range(start_step, num_steps_shown, step_size)]

    plt.gca().xaxis.set_major_locator(mticker.MultipleLocator(base=step_size, offset=start_step))
    plt.gca().xaxis.set_tick_params(labelsize=8)

    plt.plot(x_vals, [avg_diffes[i][0] * 100 for i in x_vals],
             marker="x", color="b")  # starting losses, solutions are good in the beginning!

    for i in range(start_step, num_steps_shown, step_size):
        plt.annotate(str(round(avg_diffes[i][0] * 100, 2)), (i, avg_diffes[i][0] * 100),
                     textcoords="offset points", xytext=(0, 4), fontsize=8)

    plt.xlabel("Time step")
    plt.ylabel("Initial flow deviation in %")
    plt.title(f"Solution distance using solution of previous time step")
    plt.legend()
    plt.savefig(f"out/plots/ex5_solution_distance.png")
    plt.close()

    ##### initial training (without continuation) ######

    plt.plot(avg_diffes[0][:start_iter] * 100, marker="x", markersize=2, color="b")

    plt.annotate(f"{round(avg_diffes[0][0] * 100, 2)} ", (0, avg_diffes[0][0] * 100),
                 textcoords="offset points", xytext=(-1, 0), fontsize=8)
    #plt.annotate(f"{round(losses[0][100], 2)} ", (0, losses[0][100]),
    #             textcoords="offset points", xytext=(-1, 5), fontsize=8)
    plt.annotate(f"{round(avg_diffes[0][200] * 100, 2)} ", (200, avg_diffes[0][200] * 100),
                 textcoords="offset points", xytext=(-1, 5), fontsize=8)
    plt.annotate(f"{round(avg_diffes[0][start_iter - 1] * 100, 2)} ",
                 (start_iter - 1, avg_diffes[0][start_iter - 1] * 100),
                 textcoords="offset points", xytext=(-1, 5), fontsize=8)

    plt.xlabel("Iterations")
    plt.ylabel("Mean flow deviation in %")
    plt.title(f"Training curve of first time step")
    plt.legend()
    plt.savefig(f"out/plots/ex5_first_time_step.png")
    plt.close()

    ######### continuation #########

    num_steps_shown = 5

    for i in range(1, num_steps_shown):
        color = "b"
        if i % 2 == 0:
            color = "r"
        #plt.plot(np.concatenate([losses[i][:] for i in range(0, num_steps_shown)]), marker="x", markersize=1,
        #         color=color)
        if i == 0:
            plt.plot([k for k in range(0, start_iter)], 100 * avg_diffes[i][:start_iter],
                     color=color)
        if i == 1:
            plt.plot([k for k in range(0, max_iter)], 100 * avg_diffes[i][:max_iter],
                     color=color)
            plt.plot([k for k in range(0, max_iter, 10)], 100 * avg_diffes[i][:max_iter:10], marker="x", markersize=8,
                     color=color, linestyle="none")
        if i >= 2:
            plt.plot([max_iter * (i - 1) + k for k in range(0, max_iter)], 100 * avg_diffes[i][:max_iter],
                     color=color)
            plt.plot([max_iter * (i - 1) + k for k in range(0, max_iter, 10)], 100 * avg_diffes[i][:max_iter:10],
                     marker="x", markersize=8, color=color, linestyle="none")
        # vertical line
        plt.axvline((i - 1) * 300 + 100, color="black", linestyle="--")  # at iteration 100

    """
    for i in range(0, num_steps_shown):
        if i < 2:
            plt.annotate(f"time step {i} ", (start_iter*i, -150),
                         textcoords="offset points", xytext=(0, 0), fontsize=8)
        if i >= 2:
            plt.annotate(f"time step {i} ", (start_iter + max_iter*(i-1), -150),
                        textcoords="offset points", xytext=(0, 0), fontsize=8)
    """

    """
    for i in range(num_steps_shown - 1):
        x_lower = i * max_iter + 20
        x_upper = i * max_iter + 30

        plt.vlines(x=x_lower, ymin=losses[i+1][19] - 0.5, ymax=losses[i+1][29] + 0.5, colors='black', linestyles="dotted")
        plt.vlines(x=x_upper, ymin=losses[i+1][19] - 0.5, ymax=losses[i+1][29] + 0.5, colors='black', linestyles="dotted")
    """

    plt.xlabel("Iterations")
    plt.ylabel("Flow deviations in %")

    plt.title(f"Training continuation with new injections")
    plt.legend()
    plt.savefig(f"out/plots/ex5a_time_series.png")
    plt.close()

    """
    # look at https://github.com/Grid2op/lightsim2grid/blob/master/src/batch_algorithm/TimeSeries.cpp
    from lightsim2grid import TimeSerie
    time_serie = TimeSerie(env)
    computer = time_serie.computer
    v_init = env.backend.V

    # use one of these and look into how they operate
    time_serie.compute_V(0, 0, v_init)
    status = computer.compute_Vs() # TODO this needs the injections
    """


if __name__ == "__main__":
    main()

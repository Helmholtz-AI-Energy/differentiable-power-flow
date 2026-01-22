"""
Use batching and a gpu to solve the time series in a faster way.
"""

"""
We evaluate DPF with the regards to Time Series capabilities.
We hope to see that "close" solutions need less iterations.

Idea: Make hyperparameters in 2 phases: "Search mode" and "Fine-tune mode",
e.g. use a high LR for phase 1 and a smaller LR for phase 2.
and use only the fine-tune mode for Time Series?
"""

import pickle
import matplotlib.pyplot as plt


def main():
    batch_sizes = [1, 2, 4, 8, 16, 32, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500]
    max_iters = [1000]
    use_gpus = [False]

    plot_times = True  # set true if you want to plot the speedup of batching
    if plot_times:
        for use_gpu in use_gpus:
            device = "gpu" if use_gpu else "cpu"
            times_per_iteration_per_sample = []
            # create time_per_iteration vs batch_size plot
            for batch_size in batch_sizes:
                for max_iter in max_iters:
                    with open(
                        f"out/temp/ex5b_time_series_{use_gpu}_{max_iter}_{batch_size}.pkl",
                        "rb",
                    ) as readTOFile:
                        results = pickle.load(readTOFile)
                        losses = results["losses"]
                        times = results["times"]
                        individual_losses = results["individual_losses"]
                        current_mean_time = (
                            (times[-1] - times[-101]) / 100 * 1000
                        )  # use 100 iterations to calculate the time in ms
                        times_per_iteration_per_sample.append(
                            current_mean_time / batch_size
                        )

            plt.xlabel("Batch Size")
            plt.ylabel("Time per iteration and sample (ms)")
            plt.title(f"Speed-up from batching ({device})")
            plt.legend()
            plt.plot(batch_sizes, times_per_iteration_per_sample, color="blue")
            plt.savefig(f"out/plots/ex5b_time_series_batching_{device}.png")
            plt.close()

    # plot 2: per-grid losses after joint training, TODO work
    # one plot where all the local training curves are shown together

    plot_losses = True
    if plot_losses:
        for batch_size in batch_sizes:
            for max_iter in [1000]:
                with open(
                    f"out/temp/ex5b_time_series_{use_gpu}_{max_iter}_{batch_size}.pkl",
                    "rb",
                ) as readTOFile:
                    results = pickle.load(readTOFile)
                    losses = results["losses"]
                    individual_losses = results["individual_losses"]

                    if batch_size == 1:
                        plt.title(f"Individual training")
                    else:
                        plt.title(f"Batched training of {batch_size} grids")

                    plt.xlabel("Iterations")
                    plt.ylabel("Loss")
                    for batch in range(batch_size):
                        plt.plot(individual_losses[batch, 1:100])
                    if batch_size == 1:
                        plt.plot(losses[1:100], color="blue", label="Loss")
                    else:
                        plt.plot(losses[1:100], color="blue", label="Combined loss")
                    plt.legend()
                    plt.savefig(
                        f"out/plots/ex5b_time_series_batching_losses_{batch_size}.png"
                    )
                    plt.close()


if __name__ == "__main__":
    main()

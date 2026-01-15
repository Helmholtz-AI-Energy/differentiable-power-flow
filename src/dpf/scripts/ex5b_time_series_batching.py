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

import grid2op
import torch
from lightsim2grid import LightSimBackend

from dpf.dataset import SmallTimeSeriesDataset
from dpf.solvers.solver_torch_batched import TimeSeriesPowerFlowSolverBatched



def run_experiment(max_iter, batchsize, use_gpu, evaluate_losses):
    # load data
    ts_dataset = SmallTimeSeriesDataset()
    line_status, topo_vect, Ybus, PV_nodes, slack_id = ts_dataset.get_fixed_attributes()
    inputs, targets = ts_dataset.get_injections()
    prod_p, prod_v, load_p, load_q, Sbus = inputs  # shape [time_series_length, nb_bus]
    a_or, a_ex, p_or, p_ex, v_or, v_ex, theta_or, theta_ex = targets

    env_name = "l2rpn_idf_2023"
    env = grid2op.make(env_name, backend=LightSimBackend())
    backend = env.backend

    # rounded hyperparams
    hyperparams = {
        "optimizer_class": torch.optim.Adam,
        "optimizer_kwargs": {"lr": 0.0356, "betas": (0.9802, 0.9440)},
        "scheduler_class": torch.optim.lr_scheduler.StepLR,
        "scheduler_kwargs": {"step_size": 100, "gamma": 0.773},
        "loss_fn": torch.nn.MSELoss(),
        "max_iter": max_iter,
        "tol": 1e-8}

    print(Sbus.shape[0])  # 500

    Sbus = Sbus[:batchsize, :]  # only do one batch

    print("batch shape with inactive buses: ", Sbus.shape)

    solver = TimeSeriesPowerFlowSolverBatched(backend=backend, hyperparams=hyperparams)
    solver.set_batch_size(batchsize)
    solver.set_gpu_usage(use_gpu)

    # leave it like this for now, change the Sbus in the run_time_series_batched method
    print("preprocess")
    solver.preprocess(topo_vect, prod_p, prod_v, load_p, load_q, Ybus, Sbus, PV_nodes)

    print("init voltages")
    solver.init_v("ones")

    # print("prepare fixed inputs")
    # solver.prepare_fixed_inputs()

    print("run batched time series solver")
    losses, times, individual_losses = solver.run_time_series_batched(evaluate_losses=evaluate_losses)

    # losses has shape [num_iterations]
    # and NOT: [batch_size, num_iterations] since the loss is an aggregation of all time steps simultaneously

    losses = list(losses)

    results = {
        "losses": losses,
        "times": times,
        "individual_losses": individual_losses
    }

    # print(losses)
    print(times[-1] - times[-2])


    with open(f"out/temp/ex5b_time_series_{use_gpu}_{max_iter}_{batchsize}.pkl", "wb") as writeTOFile:
        pickle.dump(results, writeTOFile)

def main():
    print("Starting experiment 5b")
    evaluate_grid_loss = True  # set to False when you want to measure time

    batch_sizes = [1, 2, 4, 8, 16, 32, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500]
    max_iters = [1000]
    use_gpus = [False]

    for batch_size in batch_sizes:
        for max_iter in max_iters:
            for use_gpu in use_gpus:
                run_experiment(max_iter=max_iter, batchsize=batch_size, use_gpu=use_gpu, evaluate_losses=evaluate_grid_loss)


if __name__ == "__main__":
    main()

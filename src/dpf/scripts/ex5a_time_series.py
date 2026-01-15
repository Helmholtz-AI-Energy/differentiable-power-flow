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
from dpf.solvers.solver_torch_time_series import TimeSeriesPowerFlowSolver


def run_experiment(start_iter, max_iter):
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
        "start_iter": start_iter,
        "tol": 1e-8}

    # tuning first step

    continuation_hyperparams = {
        "optimizer_class": torch.optim.Adam,
        "optimizer_kwargs": {"lr": 0.0002707478834355143, "betas": (0.7847655810896329, 0.6624293009401483)},
        "scheduler_class": torch.optim.lr_scheduler.ReduceLROnPlateau,
        "scheduler_kwargs": {"factor": 0.8, "patience": 2, "threshold": 0.03880181832012029,
                             "cooldown": 2},
        "loss_fn": torch.nn.MSELoss(),
        "max_iter": max_iter,
        "tol": 1e-8}

    solver = TimeSeriesPowerFlowSolver(backend=backend, hyperparams=hyperparams,
                                       continuation_hyperparams=continuation_hyperparams)
    solver.preprocess(topo_vect, prod_p[0], prod_v[0], load_p[0], load_q[0], Ybus, Sbus[0], PV_nodes)
    # the productions and loads are already present in Sbus and can be ignored for power-flows
    # S = gens - loads for the corresponding bus

    solver.init_v("ones")

    losses = solver.run_time_series(prod_p, prod_v, load_p, load_q, Sbus, freeze_start_params=False,
                                    report_metrics=True)
    average_percentage_diffes = solver.average_percentage_diffes
    # losses has shape [num_time_steps, num_iterations]

    print(losses[1:, max_iter - 1])

    losses = list(losses)
    average_percentage_diffes = list(average_percentage_diffes)

    results = {
        "losses": losses,
        "average_percentage_diffes": average_percentage_diffes
    }

    with open(f"out/temp/ex5_time_series.pkl", "wb") as writeTOFile:
        pickle.dump(results, writeTOFile)


def main():
    print("Starting experiment 5")
    start_iter = 1000
    max_iter = 300
    run_experiment(start_iter=start_iter, max_iter=max_iter)


if __name__ == "__main__":
    main()

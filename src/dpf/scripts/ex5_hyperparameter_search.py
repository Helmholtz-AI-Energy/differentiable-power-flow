"""
Hyperparameter search to find the parameters that minimize the time-series MSE over 1000 initial iterations and
20 continuation iterations.
"""

import grid2op
import torch
from lightsim2grid import LightSimBackend
import optuna
import numpy as np

from dpf.dataset import SmallTimeSeriesDataset
from dpf.solvers.solver_torch_time_series import TimeSeriesPowerFlowSolver


def objective(
    trial,
    goal="start",
    freeze_start_params=False,
    start_params=None,
    continuation_params=None,
):
    optimizer_strat = "Adam"
    scheduler_strat = "StepLR"
    start_iter = 1000  # default if not overwritten
    max_iter = 25  # default if not overwritten

    if start_params is None:
        # optimizer
        lr = trial.suggest_float("lr", 1e-8, 1e-1)  # learning rate
        beta_one = trial.suggest_float(
            "beta1", 0.1, 0.999
        )  # exponential decay rate for momentum
        beta_two = trial.suggest_float(
            "beta2", 0.1, 0.9999999
        )  # exponential decay rate for velocity
        optimizer_class = torch.optim.Adam
        optimizer_kwargs = {"lr": lr, "betas": (beta_one, beta_two)}

        # scheduler
        # step_size = trial.suggest_int("step_size", 1, 1000)
        # gamma = trial.suggest_float("gamma", 0.0001, 0.99)
        # scheduler_class = torch.optim.lr_scheduler.StepLR
        # scheduler_kwargs = {"step_size": step_size, "gamma": gamma}

        # ReduceLROnPlateau
        factor = trial.suggest_float("factor", 0.01, 0.99)
        patience = trial.suggest_int("patience", 0, 100)
        # threshold_mode = trial.suggest_categorical("threshold", ["rel", "abs"])
        threshold_mode = "rel"
        threshold = trial.suggest_float("threshold", 1e-6, 1e-1)
        cooldown = trial.suggest_int("cooldown", 0, 100)
        scheduler_class = torch.optim.lr_scheduler.ReduceLROnPlateau
        scheduler_kwargs = {
            "factor": factor,
            "patience": patience,
            "threshold_mode": threshold_mode,
            "threshold": threshold,
            "cooldown": cooldown,
        }

        hyperparams = {
            "optimizer_class": optimizer_class,
            "optimizer_kwargs": optimizer_kwargs,
            "scheduler_class": scheduler_class,
            "scheduler_kwargs": scheduler_kwargs,
            "loss_fn": torch.nn.MSELoss(),
            "start_iter": start_iter,
            "tol": 1e-8,
        }
    else:
        hyperparams = start_params
        start_iter = hyperparams["start_iter"]

    if continuation_params is None:
        if freeze_start_params:
            continuation_hyperparams = {
                "max_iter": max_iter,  # for now only deactivate scheduler to freeze previous learning rate,
            }
        else:
            # optimizer
            lr_cont = trial.suggest_float("lr_cont", 1e-6, 1e-1)  # learning rate
            beta_one_cont = trial.suggest_float(
                "beta_one_cont", 0.0001, 0.999
            )  # exponential decay rate for momentum
            beta_two_cont = trial.suggest_float(
                "beta_two_cont", 0.0001, 0.999
            )  # exponential decay rate for velocity
            optimizer_class_cont = torch.optim.Adam
            optimizer_kwargs_cont = {
                "lr": lr_cont,
                "betas": (beta_one_cont, beta_two_cont),
            }

            # scheduler, stepLR
            # step_size_cont = 10
            # step_size_cont = trial.suggest_int("step_size_cont", 1, 100)
            # gamma_cont = trial.suggest_float("gamma_cont", 0.0001, 0.99)
            # scheduler_class_cont = torch.optim.lr_scheduler.StepLR
            # scheduler_kwargs_cont = {"step_size": step_size_cont, "gamma": gamma_cont}

            # ReduceLROnPlateau
            factor_cont = trial.suggest_float("factor", 0.01, 0.99)
            patience_cont = trial.suggest_int("patience", 0, 100)
            # threshold_mode = trial.suggest_categorical("threshold", ["rel", "abs"])
            threshold_mode_cont = "rel"
            threshold_cont = trial.suggest_float("threshold", 1e-6, 1e-1)
            cooldown_cont = trial.suggest_int("cooldown", 0, 100)
            scheduler_class_cont = torch.optim.lr_scheduler.ReduceLROnPlateau
            scheduler_kwargs_cont = {
                "factor": factor_cont,
                "patience": patience_cont,
                "threshold_mode": threshold_mode_cont,
                "threshold": threshold_cont,
                "cooldown": cooldown_cont,
            }

            continuation_hyperparams = {
                "optimizer_class": optimizer_class_cont,
                "optimizer_kwargs": optimizer_kwargs_cont,
                "scheduler_class": scheduler_class_cont,
                "scheduler_kwargs": scheduler_kwargs_cont,
                "loss_fn": torch.nn.MSELoss(),
                "max_iter": max_iter,  # for now only deactivate scheduler to freeze previous learning rate,
                "tol": 1e-8,
            }
    else:
        continuation_hyperparams = continuation_params
        max_iter = continuation_hyperparams["max_iter"]

    # load data
    ts_dataset = SmallTimeSeriesDataset()
    line_status, topo_vect, Ybus, PV_nodes, slack_id = ts_dataset.get_fixed_attributes()
    inputs, targets = ts_dataset.get_injections()
    prod_p, prod_v, load_p, load_q, Sbus = inputs  # shape [time_series_length, nb_bus]
    a_or, a_ex, p_or, p_ex, v_or, v_ex, theta_or, theta_ex = targets

    env_name = "l2rpn_idf_2023"
    env = grid2op.make(env_name, backend=LightSimBackend())
    backend = env.backend

    solver = TimeSeriesPowerFlowSolver(
        backend=backend,
        hyperparams=hyperparams,
        continuation_hyperparams=continuation_hyperparams,
    )
    solver.preprocess(
        topo_vect, prod_p[0], prod_v[0], load_p[0], load_q[0], Ybus, Sbus[0], PV_nodes
    )
    # the productions and loads are already present in Sbus and can be ignored for power-flows
    # S = gens - loads for the corresponding bus

    solver.init_v("ones")

    do_only_first_time_step = False
    if goal == "start" or goal == "cumulative_start":
        do_only_first_time_step = True

    # losses = solver.run_time_series(prod_p, prod_v, load_p, load_q, Sbus, freeze_start_params=freeze_start_params,
    #                                do_only_first_time_step=do_only_first_time_step)

    losses = solver.run_time_series(
        prod_p[0:2],
        prod_v[0:2],
        load_p[0:2],
        load_q[0:2],
        Sbus[0:2],
        freeze_start_params=freeze_start_params,
        do_only_first_time_step=do_only_first_time_step,
    )

    # losses has shape [num_time_steps, num_iterations] # whereas num_iterations = max(start_iter, max_iter)

    print(losses.shape)
    if goal == "start":
        initial_loss = losses[0][start_iter - 1]
        return initial_loss

    if goal == "total":
        initial_loss = losses[0][start_iter - 1]
        continuation_loss_sum = sum(losses[1:, max_iter - 1])
        total_loss = initial_loss + continuation_loss_sum
        return total_loss

    if goal == "cumulative_start":
        # weighted average of cumulative sum and last loss
        mean_cumulative_initial_loss = sum(losses[0][0 : start_iter - 1]) / start_iter
        last_loss = losses[0][start_iter - 1]  # maybe use this as well?

        return mean_cumulative_initial_loss + last_loss

    if goal == "cumulative_total":
        mean_cumulative_initial_loss = sum(losses[0][0 : start_iter - 1]) / start_iter
        mean_cumulative_continuation_loss = np.sum(losses[1:, 0 : max_iter - 1]) / (
            max_iter * (losses.shape[0] - 1)
        )
        # adds all losses in total
        total_loss = mean_cumulative_initial_loss + mean_cumulative_continuation_loss
        return total_loss


def main():
    print("Starting experiment 5")

    goal = "total"  # "start", "total", "cumulative_start"
    freeze_start_params = False

    # fix starting parameter

    # Adam with StepLR, has large fluctuations in the beginning
    start_params = {
        "optimizer_class": torch.optim.Adam,
        "optimizer_kwargs": {"lr": 0.0356, "betas": (0.9802, 0.9440)},
        "scheduler_class": torch.optim.lr_scheduler.StepLR,
        "scheduler_kwargs": {"step_size": 100, "gamma": 0.773},
        "loss_fn": torch.nn.MSELoss(),
        "start_iter": 1000,
        "tol": 1e-8,
    }

    # ReduceLROnPlateau start, only last loss
    start_params = {
        "optimizer_class": torch.optim.Adam,
        "optimizer_kwargs": {
            "lr": 0.004482156096488414,
            "betas": (0.8395198093036706, 0.7849709438171584),
        },
        "scheduler_class": torch.optim.lr_scheduler.ReduceLROnPlateau,
        "scheduler_kwargs": {
            "factor": 0.26248539570014334,
            "patience": 37,
            "threshold": 0.01945014133603648,
            "cooldown": 97,
        },
        "loss_fn": torch.nn.MSELoss(),
        "start_iter": 1000,
        "tol": 1e-8,
    }

    # ReduceLROnPlateau start, cumulative (weighted last and cumulative)
    start_params = {
        "optimizer_class": torch.optim.Adam,
        "optimizer_kwargs": {
            "lr": 0.004159466673285678,
            "betas": (0.8269486611348338, 0.7686179771446768),
        },
        "scheduler_class": torch.optim.lr_scheduler.ReduceLROnPlateau,
        "scheduler_kwargs": {
            "factor": 0.6800157049818711,
            "patience": 30,
            "threshold": 0.08664321275906847,
            "cooldown": 43,
        },
        "loss_fn": torch.nn.MSELoss(),
        "start_iter": 1000,
        "tol": 1e-8,
    }

    """
    # no momentum reset
    continuation_hyperparams = {
        "optimizer_class": torch.optim.Adam,
        "optimizer_kwargs": {"lr": 0.012726781791966579, "betas": (0.9719197829764082, 0.9960455796004551)},
        "scheduler_class": torch.optim.lr_scheduler.StepLR,
        "scheduler_kwargs": {"step_size": 10, "gamma": 0.57251458816},  # does nothing here
        "loss_fn": torch.nn.MSELoss(),
        "max_iter": 50,
        "tol": 1e-8}
    """

    # momentum reset, Adam, StepLR
    continuation_hyperparams = {
        "optimizer_class": torch.optim.Adam,
        "optimizer_kwargs": {
            "lr": 0.0058208151664865815,
            "betas": (0.9507494030955903, 0.9975508170272224),
        },
        "scheduler_class": torch.optim.lr_scheduler.StepLR,
        "scheduler_kwargs": {
            "step_size": 10,
            "gamma": 0.9634944601118305,
        },  # does nothing here
        "loss_fn": torch.nn.MSELoss(),
        "max_iter": 100,
        "tol": 1e-8,
    }

    # ReduceLROnPLateau selected by best cumulative loss
    continuation_hyperparams = {
        "optimizer_class": torch.optim.Adam,
        "optimizer_kwargs": {
            "lr": 0.0030812144154004485,
            "betas": (0.9732523371601307, 0.9518236297535477),
        },
        "scheduler_class": torch.optim.lr_scheduler.ReduceLROnPlateau,
        "scheduler_kwargs": {
            "factor": 0.527586988165259,
            "patience": 60,
            "threshold": 0.031535230054427675,
            "cooldown": 63,
        },
        "loss_fn": torch.nn.MSELoss(),
        "max_iter": 100,
        "tol": 1e-8,
    }

    continuation_hyperparams = {
        "optimizer_class": torch.optim.Adam,
        "optimizer_kwargs": {
            "lr": 0.005001632613416399,
            "betas": (0.9807870843105879, 0.9768142196939307),
        },
        "scheduler_class": torch.optim.lr_scheduler.ReduceLROnPlateau,
        "scheduler_kwargs": {
            "factor": 0.9101855298722269,
            "patience": 89,
            "threshold": 0.010055068774012972,
            "cooldown": 84,
        },
        "loss_fn": torch.nn.MSELoss(),
        "max_iter": 100,
        "tol": 1e-8,
    }

    study = optuna.create_study()

    # use start_params=None and goal="start" to only do first time-step
    # use goal="total" and start_params="start_params" to do the remaining time steps
    study.optimize(
        lambda trial: objective(
            trial,
            goal=goal,
            freeze_start_params=freeze_start_params,
            start_params=start_params,
            continuation_params=None,
        ),
        n_trials=1000,
    )

    print(f"Best params is {study.best_params} with value {study.best_value}")

    # goal: total, max_iter: 50, start_iter:100, only exp_avg reset
    # {'lr_cont': 0.0058208151664865815,
    # 'beta_one_cont': 0.9507494030955903,
    # 'beta_two_cont': 0.9975508170272224,
    # 'gamma_cont': 0.9634944601118305}.
    # max_iter : 50 , start_iter: 1000
    # Best is trial 222 with value: 1398.751608895022.

    # ReduceLROnPlateau with cumulative weighted loss, max_iter 100, second momentum kept, first reset
    # Trial 773 finished with value: 64.0153568044116 and parameters:
    # {'lr_cont': 0.0030812144154004485,
    # 'beta_one_cont': 0.9732523371601307,
    # 'beta_two_cont': 0.9518236297535477,
    # 'factor': 0.527586988165259,
    # 'patience': 60,
    # 'threshold': 0.031535230054427675,
    # 'cooldown': 63}.
    # Best is trial 773 with value: 64.0153568044116.

    # momentum also reset, rest similar as above:
    # Trial 33 finished with value: 284.6194860391932 and parameters:
    # {'lr_cont': 0.01109789957183186,
    # 'beta_one_cont': 0.7961630876369263,
    # 'beta_two_cont': 0.8373222307034363,
    # 'factor': 0.5323406315612105,
    # 'patience': 18,
    # 'threshold': 0.08180419209512793,
    # 'cooldown': 28}. Best is trial 23 with value: 71.51193687114136.

    ######
    # again only "total", 100 iterations, ReduceLROnPlateau
    # {'lr_cont': 0.005001632613416399,
    # 'beta_one_cont': 0.9807870843105879,
    # 'beta_two_cont': 0.9768142196939307,
    # 'factor': 0.9101855298722269,
    # 'patience': 89,
    # 'threshold': 0.010055068774012972,
    # 'cooldown': 84}.
    # Best is trial 124 with value: 684.1051765279092


if __name__ == "__main__":
    main()

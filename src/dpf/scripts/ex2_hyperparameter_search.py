import os
import sys

import grid2op
import numpy as np
import optuna
import pandas as pd
import torch.optim
from lightsim2grid import LightSimBackend

from dpf.dataset import LipsDataset
from dpf.solvers.solver_torch import TorchPowerFlowSolver


def objective(
    trial,
    inputs,
    targets,
    env,
    backend,
    fixed_optimizer_name=None,
    fixed_scheduler_strat=None,
    max_iter_fixed=None,
    fixed_loss_fn=None,
):
    if fixed_optimizer_name is not None:
        optimizer_name = fixed_optimizer_name
    else:
        optimizer_name = trial.suggest_categorical(
            "optimizer", ["Adam", "RMSprop", "SGD"]
        )
    if fixed_scheduler_strat is not None:
        scheduler_strat = fixed_scheduler_strat
    else:
        scheduler_strat = trial.suggest_categorical(
            "scheduler", ["constant", "StepLR", "ReduceLROnPlateau"]
        )
    if max_iter_fixed is not None:
        max_iter = max_iter_fixed
    else:
        max_iter = 1000
    if fixed_loss_fn is not None:
        loss_fn_name = fixed_loss_fn
    else:
        loss_fn_name = trial.suggest_categorical(
            "loss_fn", ["L1", "L1Sum", "MSE", "Huber"]
        )

    (
        prod_p,
        prod_v,
        load_p,
        load_q,
        line_status,
        topo_vect,
        Ybus,
        Sbus,
        PV_nodes,
        slack,
    ) = inputs
    a_or, a_ex, p_or, p_ex, v_or, v_ex, theta_or, theta_ex = (
        targets  # maybe use these for evaluation as well?
    )

    optimizer_class = None
    optimizer_kwargs = None
    if optimizer_name == "Adam":
        lr = trial.suggest_float("lr", 1e-6, 1e-1)  # learning rate
        beta_one = trial.suggest_float(
            "beta1", 0.5, 0.999
        )  # exponential decay rate for momentum
        beta_two = trial.suggest_float(
            "beta2", 0.8, 0.9999999
        )  # exponential decay rate for velocity
        optimizer_class = torch.optim.Adam
        optimizer_kwargs = {"lr": lr, "betas": (beta_one, beta_two)}
    if optimizer_name == "SGD":
        lr = trial.suggest_float("lr", 1e-6, 1e-1)
        momentum = trial.suggest_float("momentum", 0.0, 0.99)
        dampening = trial.suggest_float("dampening", 0.0, 0.99)
        weight_decay = trial.suggest_float("weight_decay", 0.0, 0.99)
        optimizer_class = torch.optim.SGD
        optimizer_kwargs = {
            "lr": lr,
            "momentum": momentum,
            "dampening": dampening,
            "weight_decay": weight_decay,
        }
    if optimizer_name == "RMSprop":
        lr = trial.suggest_float("lr", 1e-6, 1e-1)
        alpha = trial.suggest_float("alpha", 0.1, 0.999)
        weight_decay = trial.suggest_float("weight_decay", 0.0, 0.999)
        momentum = trial.suggest_float("momentum", 0.0, 0.999)
        optimizer_class = torch.optim.RMSprop
        optimizer_kwargs = {
            "lr": lr,
            "alpha": alpha,
            "weight_decay": weight_decay,
            "momentum": momentum,
        }

    scheduler_class = None
    scheduler_kwargs = None
    if scheduler_strat == "constant":
        scheduler_class = torch.optim.lr_scheduler.ConstantLR
        scheduler_kwargs = {"factor": 1.0, "total_iters": 9999999999}
    if scheduler_strat == "StepLR":
        step_size = trial.suggest_int("step_size", 1, 100)
        gamma = trial.suggest_float("gamma", 0.0001, 0.99)
        scheduler_class = torch.optim.lr_scheduler.StepLR
        scheduler_kwargs = {"step_size": step_size, "gamma": gamma}
    if scheduler_strat == "ReduceLROnPlateau":
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
    if scheduler_strat == "MultiStepLR":
        num_milestones = trial.suggest_int("num_milestones", 1, 20)
        milestones = sorted(
            trial.suggest_int("milestone_" + str(i), 1, 300)
            for i in range(num_milestones)
        )
        gamma = trial.suggest_float("gamma", 0.0001, 0.99)
        scheduler_class = torch.optim.lr_scheduler.MultiStepLR
        scheduler_kwargs = {"milestones": milestones, "gamma": gamma}

    loss_fn = None

    if loss_fn_name == "L1":  # ["L1", "MSE", "CE"])
        loss_fn = torch.nn.L1Loss()
    if loss_fn_name == "L1Sum":
        loss_fn = torch.nn.L1Loss(reduction="sum")
    if loss_fn_name == "MSE":
        loss_fn = torch.nn.MSELoss()
    if loss_fn_name == "CE":
        loss_fn = torch.nn.CrossEntropyLoss()  # this somehow does not work
    if loss_fn_name == "Huber":
        loss_fn = torch.nn.HuberLoss()

    hyperparams = {
        "optimizer_class": optimizer_class,
        "optimizer_kwargs": optimizer_kwargs,
        "scheduler_class": scheduler_class,
        "scheduler_kwargs": scheduler_kwargs,
        # "loss_fn": torch.nn.MSELoss(),
        "loss_fn": loss_fn,
        "max_iter": max_iter,
        "tol": 1e-8,
    }

    torch_solver = TorchPowerFlowSolver(backend=backend, hyperparams=hyperparams)

    torch_solver.preprocess(
        topo_vect, prod_p, prod_v, load_p, load_q, Ybus, Sbus, PV_nodes
    )
    # torch_solver.init_v("ones")
    # torch_solver.init_v("dc")  # this might be important as a dc approximation takes some time as well
    torch_solver.init_v(
        "ones"
    )  # this might be important as a dc approximation takes some time as well

    # generate targets by running pf here
    torch_solver.run_pf(report_metrics=True)
    loss = torch_solver.eval_dict["mse"][-1]
    # loss = torch_solver.best_loss
    return loss  # last MSE loss


def run_trials(optimizer_strat, scheduler_strat, max_iter, loss_fn, n_trials):
    download_data = not os.path.isdir("data/input_data_local/lips_idf_2023")

    if download_data:
        print("downloading the lips dataset")
    else:
        print("dataset folder already exists")

    lips_dataset = LipsDataset(download_data=download_data, load_data=True)

    inputs, targets = lips_dataset.get_sample(lips_dataset.train_dataset, 0)
    env = grid2op.make(lips_dataset.benchmark.env_name, backend=LightSimBackend())
    backend = env.backend

    study = optuna.create_study()
    study.optimize(
        lambda trial: objective(
            trial,
            inputs,
            targets,
            env,
            backend,
            optimizer_strat,
            scheduler_strat,
            max_iter,
            loss_fn,
        ),
        n_trials=n_trials,
    )
    print(f"Best params is {study.best_params} with value {study.best_value}")

    df = study.trials_dataframe(attrs=("params", "value", "state"))
    df.to_csv(
        f"out/temp/ex2_trials_{optimizer_strat}_{scheduler_strat}_{max_iter}.csv",
        index=False,
    )


def main():
    n_trials = 100

    optimizer_strats = ["RMSprop", "Adam", "SGD"]
    # optimizer_strats = ["Adam"]

    scheduler_strats = ["constant", "StepLR", "ReduceLROnPlateau", "MultiStepLR"]
    # scheduler_strats = ["constant"]

    loss_func_names = ["MSE"]

    num_iters = [50, 1000]

    for optimizer_strat in optimizer_strats:
        for scheduler_strat in scheduler_strats:
            for loss_fn_name in loss_func_names:
                for num_iter in num_iters:
                    print(f"trying {optimizer_strat, scheduler_strat}")
                    run_trials(
                        optimizer_strat,
                        scheduler_strat,
                        num_iter,
                        n_trials=n_trials,
                        loss_fn=loss_fn_name,
                    )


if (
    __name__ == "__main__"
):  # run with python ex2_hyperparameter_search.py or give arguments to test a specific comb
    main()
    # best losses each:
    # Adam + constant: 0.759
    # Adam + StepLR: 0.501
    # Adam + ReduceLROnPlateau: 0.509  # with standard threshold
    # Adam + MultiStepLR: 0.71
    # SGD + constant: 1.375
    # SGD + StepLR: 0.762
    # SGD + ReduceLROnPlateau 0.77  # with standard threshold
    # SGD + MultiStepLR: 0.79
    # RMSprop + constant:  0.58
    # RMSprop + StepLR: 0.00137
    # RMSprop + ReduceLROnPlateau:  0.001534
    # RMSprop + MultiStepLR: 0.0761

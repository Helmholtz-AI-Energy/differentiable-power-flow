"""
This experiment compares the methods
a) DC-approximation
b) Newton-Raphson
c) our torch-optimization
regarding computation time and accuracy in a pareto-plot.

For our optimizer we use the hyperparameters found in ex2 and we evaluate the performance on unseen samples.
"""
import argparse
import sys
import time
import pickle
import grid2op
import torch
from lightsim2grid import LightSimBackend

from dpf.dataset import LipsDataset
from dpf.solvers.solver_torch import TorchPowerFlowSolver
from dpf.solvers.solver_newton_raphson_cpp import NRPowerFlowSolverCPP


def run_experiment(strategy="NR", sample=42, to_optimizer="RMSprop", save_results=False, seed=0):
    # load data
    lips_dataset = LipsDataset(load_data=True)
    inputs, targets = (lips_dataset.get_sample(lips_dataset.train_dataset, sample))
    prod_p, prod_v, load_p, load_q, line_status, topo_vect, Ybus, Sbus, PV_nodes, slack = inputs
    a_or, a_ex, p_or, p_ex, v_or, v_ex, theta_or, theta_ex = targets

    env = grid2op.make(lips_dataset.benchmark.env_name, backend=LightSimBackend())
    backend = env.backend

    # do the power-flow-calculation to retrieve voltages

    if strategy == "DC":
        print("DC approximation")
        # output: single time and metrics on power mismatch
        start_time = time.perf_counter()
        solver = NRPowerFlowSolverCPP(backend=backend)
        solver.preprocess(topo_vect, prod_p, prod_v, load_p, load_q, Ybus, Sbus, PV_nodes)
        solver.init_v("dc")

        """ones-init vs dc-init
        solver2 = NRPowerFlowSolverCPP(backend=backend)
        solver2.preprocess(topo_vect, prod_p, prod_v, load_p, load_q, Ybus, Sbus, PV_nodes)
        solver2.init_v("ones")
        print("mse loss ones", solver2.calculate_l2_loss())
        print("mse loss dc", solver.calculate_l2_loss())
        exit()
        # mse loss ones 4169.617663039935
        # mse loss dc 9214.154891490241
        """

        V = solver.V
        end_time = time.perf_counter()

        times = []
        voltage = []
        mismatches_mse = []

        times.append(end_time - start_time)
        voltage.append(V)
        mismatches_mse.append(solver.calculate_l2_loss())

        results = {
            "times": times,
            "voltage": voltage,
            "mismatches": mismatches_mse,
        }
        if save_results:
            with open(f"out/temp/ex3_DC_{seed}.pkl", "wb") as writeDCFile:
                pickle.dump(results, writeDCFile)

    if strategy == "NR":
        # output: list of times/voltages/mismatches each iteration
        print("NR")
        # output: single time and metrics on power mismatch
        start_time = time.perf_counter()
        solver = NRPowerFlowSolverCPP(backend=backend)
        solver.preprocess(topo_vect, prod_p, prod_v, load_p, load_q, Ybus, Sbus, PV_nodes)
        solver.init_v("dc")
        solver.run_pf()
        V = solver.V
        end_time = time.perf_counter()

        times = []
        voltage = []
        mismatches_mse = []
        times.append(end_time - start_time)
        voltage.append(V)
        mismatches_mse.append(solver.calculate_l2_loss())

        results = {
            "times": times,
            "voltage": voltage,
            "mismatches": mismatches_mse,
        }

        if save_results:
            with open(f"out/temp/ex3_NR_{seed}.pkl", "wb") as writeNRFile:
                pickle.dump(results, writeNRFile)

    if strategy == "TO":
        # output: list of times/voltages/mismatches each iteration

        start_time = time.perf_counter()

        hyperparams = None
        if to_optimizer == "RMSprop":
            hyperparams = {
                "optimizer_class": torch.optim.RMSprop,
                "optimizer_kwargs": {"lr": 0.00605328655257, "alpha": 0.5545244691705359,
                                     "weight_decay": 0.0562911935674707, "momentum": 0.9805599222530874},
                "scheduler_class": torch.optim.lr_scheduler.StepLR,
                "scheduler_kwargs": {"gamma": 0.9445849898428068, "step_size": 6},
                "loss_fn": torch.nn.MSELoss(),
                "max_iter": 1000,
                "tol": 1e-8
            }

        if to_optimizer == "Adam":
            hyperparams = {
                "optimizer_class": torch.optim.Adam,
                "optimizer_kwargs": {"lr": 0.003377, "betas": (0.979681, 0.963442)},
                "scheduler_class": torch.optim.lr_scheduler.ReduceLROnPlateau,
                "scheduler_kwargs": {"factor": 0.547191, "patience": 41, "threshold_mode": "rel",
                                     "threshold": 0.067321, "cooldown": 97},
                "loss_fn": torch.nn.MSELoss(),
                "max_iter": 1000,
                "tol": 1e-8}

        solver = TorchPowerFlowSolver(backend=backend, hyperparams=hyperparams)
        solver.preprocess(topo_vect, prod_p, prod_v, load_p, load_q, Ybus, Sbus, PV_nodes)

        # solver.init_v("uniform_complex", random_init_seed=seed)
        solver.init_v("ones")

        time_before_pf = time.perf_counter()
        solver.run_pf()
        V = solver.V  # best voltage reachable
        end_time = time.perf_counter()

        times = [x + (time_before_pf - start_time) for x in solver.times_list]
        times_without_opt_init = [x - solver.optimizer_init_time for x in times]
        times_without_opt_init.append(end_time - start_time - solver.optimizer_init_time)
        voltage = [V]
        mismatches_mse = list(solver.loss_list)

        mismatches_mse.append(solver.calculate_l2_loss())
        times.append(end_time - start_time)

        results = {
            "times": times,
            "voltage": voltage,
            "mismatches": mismatches_mse,
            "times_without_opt_init": times_without_opt_init
        }
        if save_results:
            with open(f"out/temp/ex3_TO_{to_optimizer}_{seed}.pkl", "wb") as writeTOFile:
                pickle.dump(results, writeTOFile)


def main(argv=None):
    print("Starting experiment 3")
    #  Do experiments twice for warm starts, once for cold starts. Comment out other methods for better reproducibility.

    if argv is None:
        argv = sys.argv[1:]

    parser = argparse.ArgumentParser(prog="ex3")
    parser.add_argument("--method", type=str, default="TO")

    args = parser.parse_args(argv)
    method = args.method.upper()
    print("using method", method)

    # method = "TO"  # "NR", "TO". "DC"

    to_optimizer = "Adam"  # "RMSprop", "Adam"
    number_of_seeds = 10

    for run_number in range(number_of_seeds):
        sample = run_number  # grid sample is the seed number
        run_experiment(method, sample, to_optimizer, save_results=False, seed=run_number)  # ensure warm start
        run_experiment(method, sample, to_optimizer, save_results=True, seed=run_number)


if __name__ == "__main__":
    main()

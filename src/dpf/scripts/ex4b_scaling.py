"""
In ex4_scaling.py the loading time is the main factor. Here, we want to find a use case where the loading is
not the limiting factor, e.g. look at
a) Ignore the loading time for now
b) Time Series
c) Cascading Failure Analysis
"""

import os
import time
import pickle
import numpy
import pandapower as pp

import numpy as np
import torch

from dpf.dataset import CustomGridDataset
from dpf.scripts.ex4_data_generation import make_grid2op_env, get_loads_gens
from dpf.solvers.solver_torch import TorchPowerFlowSolver
from dpf.solvers.solver_newton_raphson_cpp import NRPowerFlowSolverCPP


def run_experiment(strategy="NR", sample=2, to_optimizer="RMSprop", case_name="case2848pegase", use_gpu=False,
                   save_result=False):
    # load data
    custom_grid_dataset = CustomGridDataset(env_name=case_name)
    inputs, targets = custom_grid_dataset.get_sample(sample)
    prod_p, prod_v, load_p, load_q, line_status, topo_vect, Ybus, Sbus, PV_nodes, slack = inputs
    # a_or, a_ex, p_or, p_ex, v_or, v_ex, theta_or, theta_ex = targets

    # create grid2op env and the backend
    case_path = os.path.join("data/ex4_cases", case_name)
    if not os.path.exists(case_path):
        import pandapower.networks as pn

        case = getattr(pn, case_name.split(".")[0])()
        pp.to_json(case, case_path)

    # load the case file
    case = pp.from_json(case_path)
    pp.runpp(case)  # for slack

    # extract reference data
    load_p_init = 1.0 * case.load["p_mw"].values
    load_q_init = 1.0 * case.load["q_mvar"].values
    gen_p_init = 1.0 * case.gen["p_mw"].values
    sgen_p_init = 1.0 * case.sgen["p_mw"].values
    prng = np.random.default_rng(42)

    # simulate the data
    load_p_, load_q_, gen_p_, sgen_p_ = get_loads_gens(load_p_init, load_q_init, gen_p_init, sgen_p_init, prng)
    nb_ts = gen_p_.shape[0]
    # add slack !
    slack_gens = np.zeros((nb_ts, case.ext_grid.shape[0]))
    if "res_ext_grid" in case:
        slack_gens += np.tile(case.res_ext_grid["p_mw"].values.reshape(1, -1), (nb_ts, 1))
    gen_p_g2op = np.concatenate((gen_p_, slack_gens), axis=1)

    env = make_grid2op_env(case,
                           case_name,
                           load_p_,
                           load_q_,
                           gen_p_g2op,
                           sgen_p_)

    backend = env.backend

    # do the power-flow-calculation to retrieve voltages

    num_repetitions = 10

    if strategy == "DC":

        times_list = []
        for _ in range(num_repetitions):
            # output: single time and metrics on power mismatch
            solver = NRPowerFlowSolverCPP(backend=backend)
            solver.preprocess(topo_vect, prod_p, prod_v, load_p, load_q, Ybus, Sbus, PV_nodes)
            start_time = time.perf_counter()
            solver.init_v("dc")
            V = solver.V
            end_time = time.perf_counter()
            print("DC time: ", end_time - start_time)

            times = []
            voltage = []
            mismatches_mse = []

            times.append(end_time - start_time)
            voltage.append(V)
            mismatches_mse.append(solver.calculate_l2_loss())

            times_list.append(times)

        times_list = numpy.array(times_list)
        mean_times = [sum(values) / len(values) for values in zip(*times_list)]
        mean_times = list(mean_times)

        results = {
            "times": mean_times,
            "voltage": voltage,
            "mismatches": mismatches_mse,
        }

        if save_result:
            with open(f"out/temp/ex4b_DC_{case_name}_{use_gpu}.pkl", "wb") as writeDCFile:
                pickle.dump(results, writeDCFile)

    if strategy == "NR":
        # output: list of times/voltages/mismatches each iteration
        # output: single time and metrics on power mismatch
        times_list = []
        for _ in range(num_repetitions):
            solver = NRPowerFlowSolverCPP(backend=backend)
            solver.preprocess(topo_vect, prod_p, prod_v, load_p, load_q, Ybus, Sbus, PV_nodes)
            start_time = time.perf_counter()
            solver.init_v("dc")
            solver.run_pf()
            V = solver.V
            end_time = time.perf_counter()
            print("NR time: ", end_time - start_time)

            times = []
            voltage = []
            mismatches_mse = []
            times.append(end_time - start_time)
            voltage.append(V)
            mismatches_mse.append(solver.calculate_l2_loss())

            times_list.append(times)

        times_list = numpy.array(times_list)
        mean_times = [sum(values) / len(values) for values in zip(*times_list)]
        mean_times = list(mean_times)

        results = {
            "times": mean_times,
            "voltage": voltage,
            "mismatches": mismatches_mse,
        }
        if save_result:
            with open(f"out/temp/ex4b_NR_{case_name}_{use_gpu}.pkl", "wb") as writeNRFile:
                pickle.dump(results, writeNRFile)

    if strategy == "TO":
        # output: list of times/voltages/mismatches each iteration

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

        times_list = []
        for _ in range(num_repetitions):
            solver = TorchPowerFlowSolver(backend=backend, hyperparams=hyperparams)

            solver.set_gpu_usage(use_gpu=use_gpu)
            device = solver.device

            solver.preprocess(topo_vect, prod_p, prod_v, load_p, load_q, Ybus, Sbus, PV_nodes)
            start_time = time.perf_counter()
            solver.init_v("ones")
            time_before_pf = time.perf_counter()
            solver.run_pf(checkpointing=False)
            V = solver.V  # best voltage reachable
            end_time = time.perf_counter()
            print("TO time: ", end_time - start_time)

            times = [x + (time_before_pf - start_time) for x in solver.times_list]
            mismatches_mse = list(solver.loss_list)

            mismatches_mse.append(solver.calculate_l2_loss())
            times.append(end_time - start_time)

            times_list.append(times)

        times_list = numpy.array(times_list)
        mean_times = [sum(values) / len(values) for values in zip(*times_list)]
        mean_times = list(mean_times)

        results = {
            "times": mean_times,
            "mismatches": mismatches_mse,
        }

        if save_result:
            with open(f"out/temp/ex4b_TO_{to_optimizer}_{case_name}_{use_gpu}.pkl",
                      "wb") as writeTOFile:
                pickle.dump(results, writeTOFile)


def main():
    import warnings

    warnings.filterwarnings("ignore")

    print("Starting experiment 4b")

    sample = 2
    use_gpu = False
    case_names = ["case118", "case_illinois200", "case300", "case1354pegase", "case1888rte",
                  "case2869pegase", "case3120sp", "case6495rte", "case6515rte", "case9241pegase"]
    methods = ["DC", "NR", "TO"]
    to_optimizers = ["Adam"]

    grid_sizes = [118, 200, 300, 1354, 1888, 2869, 3120, 6495, 6515, 9241]

    # run experiments by running this

    run_experiment("DC", sample, "None", "case118", use_gpu=use_gpu, save_result=False)  # warm start

    for method in methods:
        for case_name in case_names:
            if method == "TO":
                for to_optimizer in to_optimizers:
                    run_experiment(method, sample, to_optimizer, case_name, use_gpu=use_gpu, save_result=True)
            else:
                run_experiment(method, sample, "None", case_name, use_gpu=use_gpu, save_result=True)


if __name__ == "__main__":
    main()

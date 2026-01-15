from dpf.scripts.ex4_data_generation import make_grid2op_env, get_loads_gens
import pandapower as pp

import os
import pickle
import matplotlib.pyplot as plt

import numpy as np
import torch

from dpf.dataset import CustomGridDataset
from dpf.solvers.solver_torch_batched import TimeSeriesPowerFlowSolverBatched


def run_experiment(max_iter, batchsize, use_gpu, evaluate_losses, strategy, strategy_amount_param):

    sample = 2
    case_name = "case9241pegase"
    custom_grid_dataset = CustomGridDataset(env_name=case_name)

    inputs, targets = custom_grid_dataset.get_sample(sample)
    prod_p, prod_v, load_p, load_q, line_status, topo_vect, Ybus, Sbus, PV_nodes, slack = inputs
    a_or, a_ex, p_or, p_ex, v_or, v_ex, theta_or, theta_ex = targets  # just ignore these for now

    # copy same values for concatenation

    # print(prod_p.shape)  # (1445,)
    prod_p = np.tile(prod_p, (batchsize, 1))
    # print(prod_p.shape)  # (batchsize,1445)

    prod_v = np.tile(prod_v, (batchsize, 1))
    load_p = np.tile(load_p, (batchsize, 1))
    load_q = np.tile(load_q, (batchsize, 1))
    Sbus = np.tile(Sbus, (batchsize, 1))

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



    # run powerflows!

    hyperparams = {
        "optimizer_class": torch.optim.Adam,
        "optimizer_kwargs": {"lr": 0.0001, "betas": (0.979681, 0.963442)},
        "scheduler_class": torch.optim.lr_scheduler.ReduceLROnPlateau,
        "scheduler_kwargs": {"factor": 0.547191, "patience": 41, "threshold_mode": "rel",
                             "threshold": 0.067321, "cooldown": 97},
        "loss_fn": torch.nn.MSELoss(),
        "max_iter": 1000,
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
    # losses, times, individual_losses = solver.run_time_series_batched(evaluate_losses=evaluate_losses)
    losses, times, individual_losses = solver.run_pf_super_grid(evaluate_losses=evaluate_losses, strategy=strategy,
                                              strategy_amount_param=strategy_amount_param)

    results = {
        "times": times,
    }

    # print(losses)
    print(times[-1] - times[-2])


    with open(f"differentiable_powerflow/out/ex9b_{use_gpu}_{max_iter}_{batchsize}_{strategy}_{strategy_amount_param}.pkl", "wb") as writeTOFile:
        pickle.dump(results, writeTOFile)


def main():
    print("Starting experiment 9b")
    evaluate_grid_loss = False  # set to False when you want to measure time
    batch_sizes = [1, 2, 4, 8, 16, 20, 32, 40, 60, 64, 80, 100, 120, 128]
    batch_sizes = [1, 2, 4, 8, 16, 20, 32, 40, 60, 64, 80, 100, 120, 256]
    batch_sizes = [200]

    inferred_grid_sizes = [batch_size * 9241 for batch_size in batch_sizes]
    print(inferred_grid_sizes)

    max_iters = [1000]
    use_gpu = False

    #strategy = "no_connections"
    #strategy_amount_param = 0

    strategy = "total_random"  # random connections anywhere
    strategy_amount_param = 20  # nb new connections = strategy_amount_param * batch_size

    # strategy = "linear_random"  # random connections between neighboring diagonal blocks
    # strategy_amount_param = 3

    # strategy = "pairwise_random"  # random connections between every block with every block
    # strategy_amount_param = 3


    for batch_size in batch_sizes:
        for max_iter in max_iters:
            run_experiment(max_iter=max_iter, batchsize=batch_size, use_gpu=use_gpu, evaluate_losses=evaluate_grid_loss,
                 strategy=strategy, strategy_amount_param=strategy_amount_param)

    batch_sizes_to_report = [1, 2, 4, 8, 16, 20, 32, 40, 60, 64, 80, 100, 120, 128]
    batch_sizes_to_report = [1, 2, 4, 8, 16,32, 64, 128, 200, 256]

    device = "gpu" if use_gpu else "cpu"

    to_times = []
    for batch_size in batch_sizes_to_report:
        with open(f"differentiable_powerflow/out/ex9b_{use_gpu}_{max_iter}_{batch_size}_{strategy}_{strategy_amount_param}.pkl", "rb") as readFile:
            results = pickle.load(readFile)
            to_times.append(results["times"])

    iterations_to_report = [250, 500, 750, 1000]

    for (j, iteration) in enumerate(iterations_to_report):  # list of iterations to report
        if j == 0:
            plt.plot(batch_sizes_to_report, [x[iteration-1] for x in to_times], label="DPF (#iterations)", color="green",
                     marker="^", markersize=3)
        plt.plot(batch_sizes_to_report, [x[iteration-1] for x in to_times], color="green", marker="^", markersize=3)

    for (j, iteration) in enumerate(iterations_to_report):
        plt.annotate("(" + str(iteration) + ")", (batch_sizes_to_report[-1], to_times[-1][iteration-1]),
                     textcoords="offset points", xytext=(-1, 2), fontsize=9)

    plt.xlabel("Grid Size in multiples of 9241")
    plt.ylabel("Time in s")
    plt.title(f"Supergrid runtime using DPF on {device}")
    plt.legend()
    # plt.show()
    plt.savefig(f"out/temp/ex9b_scalability_{use_gpu}_{strategy}_{strategy_amount_param}.png")
    plt.close()

if __name__ == "__main__":
    main()



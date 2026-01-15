from dpf.scripts.ex4_data_generation import make_grid2op_env, get_loads_gens
import pandapower as pp

import os
import pickle
import matplotlib.pyplot as plt

import numpy as np
import torch

from dpf.dataset import CustomGridDataset
from dpf.solvers.solver_torch_batched import TimeSeriesPowerFlowSolverBatched


def run_experiment(max_iter, batchsize, use_gpu, evaluate_losses):
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

    with open(f"out/temp/ex6b_{use_gpu}_{max_iter}_{batchsize}.pkl", "wb") as writeTOFile:
        pickle.dump(results, writeTOFile)


def main():
    print("Starting experiment 6b")
    evaluate_grid_loss = False  # set to False when you want to measure time
    batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
    inferred_grid_sizes = [batch_size * 9241 for batch_size in batch_sizes]
    print(inferred_grid_sizes)

    max_iters = [30]
    use_gpu = False

    for batch_size in batch_sizes:
        for max_iter in max_iters:
            run_experiment(max_iter=max_iter, batchsize=batch_size, use_gpu=use_gpu, evaluate_losses=evaluate_grid_loss)


if __name__ == "__main__":
    main()

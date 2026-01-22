from dpf.solvers.solver_torch import TorchPowerFlowSolver
from dpf.scripts.ex4_data_generation import make_grid2op_env, get_loads_gens
import pandapower as pp

import os
import time
import pickle

import numpy as np
import torch

from dpf.dataset import CustomGridDataset


def run_experiment(nb_extra_connections=0, max_iter=1000):
    sample = 2
    case_name = "case9241pegase"
    custom_grid_dataset = CustomGridDataset(env_name=case_name)

    inputs, targets = custom_grid_dataset.get_sample(sample)
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
        targets  # just ignore these for now
    )

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
    load_p_, load_q_, gen_p_, sgen_p_ = get_loads_gens(
        load_p_init, load_q_init, gen_p_init, sgen_p_init, prng
    )
    nb_ts = gen_p_.shape[0]
    # add slack !
    slack_gens = np.zeros((nb_ts, case.ext_grid.shape[0]))
    if "res_ext_grid" in case:
        slack_gens += np.tile(
            case.res_ext_grid["p_mw"].values.reshape(1, -1), (nb_ts, 1)
        )
    gen_p_g2op = np.concatenate((gen_p_, slack_gens), axis=1)

    env = make_grid2op_env(case, case_name, load_p_, load_q_, gen_p_g2op, sgen_p_)

    backend = env.backend

    print(Sbus.shape[0])  # 500

    print("batch shape with inactive buses: ", Sbus.shape)

    # Init and Preprocess first to reduce Ybus to only active buses

    hyperparams = {
        "optimizer_class": torch.optim.RMSprop,
        "optimizer_kwargs": {
            "lr": 0.00012283,
            "alpha": 0.86468552,
            "weight_decay": 0.15580241,
            "momentum": 0.99858620,
        },
        "scheduler_class": torch.optim.lr_scheduler.ReduceLROnPlateau,
        "scheduler_kwargs": {
            "factor": 0.4737,
            "patience": 40,
            "threshold_mode": "rel",
            "threshold": 0.091412,
            "cooldown": 80,
        },
        "loss_fn": torch.nn.MSELoss(),
        "max_iter": max_iter,
        "tol": 1e-8,
    }

    solver = TorchPowerFlowSolver(backend=backend, hyperparams=hyperparams)

    solver.preprocess(topo_vect, prod_p, prod_v, load_p, load_q, Ybus, Sbus, PV_nodes)

    print("Original shape: ", solver.Ybus_solver.shape)
    print("Original nnz: ", solver.Ybus_solver.nnz)
    print(type(solver.Ybus_solver))

    mean_ybus = solver.Ybus_solver.data.mean()
    std_ybus = solver.Ybus_solver.data.std()  # maybe not use this?

    print(f"adding {nb_extra_connections} many extra connections")
    solver.add_new_random_connections_to_ybus(
        nb_extra_connections, mean=mean_ybus, std=std_ybus
    )
    print(f"new shape: {solver.Ybus_solver.shape}")
    print("new nnz: ", solver.Ybus_solver.nnz)

    solver.init_v("dc")
    print("dc init finished")

    time_before = time.perf_counter()
    solver.run_pf()

    time_after = time.perf_counter()
    print("DPF time, solver only: ", time_after - time_before)

    results = {
        "times": [time_after - time_before],
    }

    # original 6a experiment - only ybus-blockdiagonal here
    # with open(f"differentiable_powerflow/out/ex6a_{batchsize}.pkl", "wb") as writeTOFile:
    #    pickle.dump(results, writeTOFile)

    with open(
        f"out/temp/ex7b_{nb_extra_connections}_{max_iter}.pkl", "wb"
    ) as writeTOFile:
        pickle.dump(results, writeTOFile)


def main():
    print("Starting experiment 7b")

    max_iter = 1000  # for case9241pegase this is enough, for random inputs more might be needed

    # nbs_extra_connections = [500, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000]
    # nbs_extra_connections = [0, 0, 10, 100, 1000, 10000, 100000]
    nbs_extra_connections = [20000, 20000, 40000, 60000, 80000, 100000]

    for nb_extra_connections in nbs_extra_connections:
        run_experiment(nb_extra_connections=nb_extra_connections, max_iter=max_iter)


if __name__ == "__main__":
    main()

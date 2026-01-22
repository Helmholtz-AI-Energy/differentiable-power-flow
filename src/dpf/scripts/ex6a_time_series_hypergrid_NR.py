from dpf.solvers.solver_newton_raphson_cpp import NRPowerFlowSolverCPP
from dpf.scripts.ex4_data_generation import make_grid2op_env, get_loads_gens
import pandapower as pp
import os
import pickle
import matplotlib.pyplot as plt

import numpy as np

from dpf.dataset import CustomGridDataset


def run_experiment(
    batchsize, ybus_scaling_method="block_diagonal", density=0.5, max_iter=6
):
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
    # Sbus = Sbus[:batchsize, :]

    print("batch shape with inactive buses: ", Sbus.shape)

    solver = NRPowerFlowSolverCPP(backend=backend)
    solver.preprocess(topo_vect, prod_p, prod_v, load_p, load_q, Ybus, Sbus, PV_nodes)
    solver.init_v("dc")
    print("dc init finished")
    solver_runtime = solver.run_pf_batched(
        batchsize,
        ybus_scaling_method=ybus_scaling_method,
        density=density,
        max_iteration_=max_iter,
    )
    V = solver.V
    print("NR time, solver only: ", solver_runtime)

    results = {
        "times": [solver_runtime],
    }

    with open(
        f"out/temp/ex6a_{batchsize}_{ybus_scaling_method}_{density}.pkl", "wb"
    ) as writeTOFile:
        pickle.dump(results, writeTOFile)


def main():
    print("Starting experiment 6a")

    # setting 1
    batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
    ybus_scaling_method = "block_diagonal"  # "block_diagonal" "random"
    density = 0.5  # this parameter is ignored for block_diagonal
    max_iter = (
        6  # for case9241pegase this is enough, for random inputs more might be needed
    )

    # 1 NR time, solver only:  0.023686250089667737 , 6 iters until convergence
    # 2 NR time, solver only:  0.045249792048707604, 6
    # 4 NR time, solver only:  0.08887624996714294, 6
    # 8 NR time, solver only:  0.1757486250717193, 6
    # 16 NR time, solver only:  0.3520847089821473, 6
    # 32 NR time, solver only:  0.7849822089774534, 6
    # 64 NR time, solver only:  1.4978153340052813, 6
    # 128 NR time, solver only:  3.035408457973972, 6
    # 256 NR time, solver only:  6.253184333909303, 6
    # 512 NR time, solver only:  13.509183832909912, 6

    for batch_size in batch_sizes:
        run_experiment(
            batchsize=batch_size,
            ybus_scaling_method=ybus_scaling_method,
            density=density,
            max_iter=max_iter,
        )


if __name__ == "__main__":
    main()

"""
In this experiment we analyze the node scaling behaviour of NR vs DPF (7a/7b are edge scaling in comparison).
For this we see two different approaches:
1) Use the case9241pegase grid clone itself K times and add connections.
   Possible connection designs to consider:
   a) Use random nodes or use a select few "important" substations
   b) Connect sub-grids all-to-all, in a line or geometrically (2d-neighborhood)

2) Select different grids (e.g. case9241pegase with another, smaller one)
    Connection structure: "important" substations, manually created, 2d-neighborhood...

In this experiment we do 1) with random connections. 2) is more beautiful as it is resembles different grids connecting
into a global grid possible but there is no new insights regarding the properties of different power-flow methods.
(in theory doable: each grid has its own power-flow-solver class to do the preprocessing,
the necessary variables Ybus, Sbus, pv, pq etc. must be collected and merged like in batching and
the concatenated Ybus-matrix needs handcrafted connections.)
"""

from dpf.solvers.solver_newton_raphson_cpp import NRPowerFlowSolverCPP
from dpf.scripts.ex4_data_generation import make_grid2op_env, get_loads_gens
import pandapower as pp

import os
import pickle

import numpy as np

from dpf.dataset import CustomGridDataset


def run_experiment(
    batchsize, max_iter=6, strategy="total_random", strategy_amount_param=0
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
    print("batch shape with inactive buses: ", Sbus.shape)

    solver = NRPowerFlowSolverCPP(backend=backend)
    solver.preprocess(topo_vect, prod_p, prod_v, load_p, load_q, Ybus, Sbus, PV_nodes)
    solver.init_v("dc")
    print("dc init finished")

    # TODO change Ybus here to a batched version and run a normal powerflow (not batched)

    solver_runtime = solver.run_pf_super_grid(
        batchsize,
        max_iteration_=max_iter,
        strategy=strategy,
        strategy_amount_param=strategy_amount_param,
    )
    V = solver.V
    print("NR time, solver only: ", solver_runtime)

    results = {
        "times": [solver_runtime],
    }

    with open(
        f"out/temp/ex9a_{batchsize}_{strategy}_{strategy_amount_param}.pkl", "wb"
    ) as writeTOFile:
        pickle.dump(results, writeTOFile)


def main():
    print("Starting experiment 9a")

    batch_sizes = [1, 2, 4, 8, 16, 20, 32, 40, 60, 64, 80, 100, 120, 128, 256]
    batch_sizes = [200]

    max_iter = (
        6  # for case9241pegase this is enough, for random inputs more might be needed
    )

    # strategy = "no_connections"
    # strategy_amount_param = 0

    strategy = "total_random"  # random connections anywhere
    strategy_amount_param = (
        20  # nb new connections = strategy_amount_param * batch_size
    )

    # strategy = "linear_random"  # random connections between neighboring diagonal blocks
    # strategy_amount_param = 3

    # strategy = "pairwise_random"  # random connections between every block with every block
    # strategy_amount_param = 3

    for batch_size in batch_sizes:
        run_experiment(
            batchsize=batch_size,
            max_iter=max_iter,
            strategy=strategy,
            strategy_amount_param=strategy_amount_param,
        )


if __name__ == "__main__":
    main()

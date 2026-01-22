"""
This experiment compares the methods
a) DC-approximation
b) Newton-Raphson
c) our torch-optimization
regarding computation time and accuracy in a pareto-plot.

For our optimizer we use the hyperparameters found in ex2 and we evaluate the performance on unseen samples.
"""

import os
import pandapower as pp
import matplotlib.pyplot as plt

import numpy as np
import torch

from dpf.dataset import CustomGridDataset
from dpf.scripts.ex4_data_generation import make_grid2op_env, get_loads_gens
from dpf.solvers.solver_torch import TorchPowerFlowSolver


def run_experiment(hyperparams):
    sample = 2
    case_name = "case9241pegase"
    # case_name = "case_illinois200"
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
    a_or, a_ex, p_or, p_ex, v_or, v_ex, theta_or, theta_ex = targets

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

    # do the power-flow-calculation to retrieve voltages

    solver = TorchPowerFlowSolver(backend=backend, hyperparams=hyperparams)
    solver.preprocess(topo_vect, prod_p, prod_v, load_p, load_q, Ybus, Sbus, PV_nodes)
    solver.init_v("ones")
    solver.run_pf()
    mismatches_mse = list(solver.loss_list)
    mismatches_mse.append(solver.calculate_l2_loss())
    print("hyperparams", hyperparams)
    print("final % deviation", 100 * mismatches_mse[-1])

    plt.plot(mismatches_mse)
    plt.savefig("out/plots/ex4_large_grid_hyperparam_search.png")

    # plt.show()


def main():
    print("Using (manual) hyperparameters to evaluate the pegase9241 grid!")
    # manual!

    # smooth curve, but last accuracy is not good
    hyperparams = {
        "optimizer_class": torch.optim.Adam,
        "optimizer_kwargs": {"lr": 0.0001, "betas": (0.979681, 0.963442)},
        "scheduler_class": torch.optim.lr_scheduler.ReduceLROnPlateau,
        "scheduler_kwargs": {
            "factor": 0.547191,
            "patience": 41,
            "threshold_mode": "rel",
            "threshold": 0.067321,
            "cooldown": 97,
        },
        "loss_fn": torch.nn.MSELoss(),
        "max_iter": 1000,
        "tol": 1e-8,
    }

    run_experiment(hyperparams)


if __name__ == "__main__":
    main()

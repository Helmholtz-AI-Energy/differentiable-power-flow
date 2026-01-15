# This experiment tests DPF under sparse and dense Ybus-matrices.
import os

import torch.nn
import numpy as np
from lightsim2grid.lightSimBackend import LightSimBackend
import grid2op

from dpf.solvers.solver_torch import TorchPowerFlowSolver
from dpf.dataset import LipsDataset
from dpf.solvers.solver_torch_dense import TorchPowerFlowSolverDense


def main():
    download_data = not os.path.isdir("data/input_data_local/lips_idf_2023")

    if download_data:
        print("downloading the lips dataset")
    else:
        print("dataset folder already exists")

    lips_dataset = LipsDataset(download_data=download_data, load_data=True)

    inputs, targets = (lips_dataset.get_sample(lips_dataset.train_dataset, 101))
    prod_p, prod_v, load_p, load_q, line_status, topo_vect, Ybus, Sbus, PV_nodes, slack = inputs
    a_or, a_ex, p_or, p_ex, v_or, v_ex, theta_or, theta_ex = targets

    #print(Ybus.shape)
    #N = int(np.sqrt(Ybus.shape[1]))
    #print(N)
    #print(Ybus.todense().reshape(N, N))

    env = grid2op.make(lips_dataset.benchmark.env_name, backend=LightSimBackend())
    backend = env.backend

    hyperparams = {
        "optimizer_class": torch.optim.Adam,
        "optimizer_kwargs": {"lr": 0.003377, "betas": (0.979681, 0.963442)},
        "scheduler_class": torch.optim.lr_scheduler.ReduceLROnPlateau,
        "scheduler_kwargs": {"factor": 0.547191, "patience": 41, "threshold_mode": "rel",
                             "threshold": 0.067321, "cooldown": 97},
        "loss_fn": torch.nn.MSELoss(),
        "max_iter": 1000,
        "tol": 1e-8}

    times = []
    nb_trials = 10
    sparsity_strategy = "sparse"
    for trial in range(nb_trials):
        if sparsity_strategy == "dense":
            torch_solver = TorchPowerFlowSolverDense(backend=backend, hyperparams=hyperparams) # dense
        if sparsity_strategy == "sparse":
            torch_solver = TorchPowerFlowSolver(backend=backend, hyperparams=hyperparams)  # sparse

        torch_solver.preprocess(topo_vect, prod_p, prod_v, load_p, load_q, Ybus, Sbus, PV_nodes)
        torch_solver.init_v("ones")  # this might be important as a dc approximation takes some time as well
        V_init_solver = torch_solver.V  # redundant input as we use the angle/magnitude representation
        torch_solver.run_pf(report_metrics=False, save_voltages=False)
        times_list = torch_solver.times_list
        time_per_pf = (times_list[-1] - times_list[-101]) / 100
        times.append(time_per_pf)

        print(time_per_pf)

    print(np.mean(times), np.std(times))

    # sparse: 0.00019116775086149574 1.6958385074158006e-06
    # dense: 0.00016866212408058346 2.1090625836096903e-06   --> DENSE IS FASTER FOR IEEE118


if __name__ == "__main__":
    main()

import os
import time

import torch.nn
import numpy as np
from lightsim2grid.lightSimBackend import LightSimBackend
import grid2op

from dpf.solvers.solver_torch import TorchPowerFlowSolver
from dpf.dataset import LipsDataset

import matplotlib.pyplot as plt


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

    torch_solver = TorchPowerFlowSolver(backend=backend, hyperparams=hyperparams)

    torch_solver.preprocess(topo_vect, prod_p, prod_v, load_p, load_q, Ybus, Sbus, PV_nodes)
    torch_solver.init_v("ones")

    V_init_solver = torch_solver.V  # redundant input as we use the angle/magnitude representation

    # inputs in solver form
    Va_solver = np.angle(V_init_solver)
    Vm_solver = np.abs(V_init_solver)
    Ybus_solver = torch_solver.Ybus_solver
    Sbus_solver = torch_solver.Sbus_solver
    slack_ids_solver = torch_solver.slack_ids_solver
    slack_id_solver = slack_ids_solver[0]  # single slack
    pv_nodes_solver = torch_solver.pv_nodes_solver
    pq_nodes_solver = torch_solver.pq_nodes_solver

    # generate targets by running pf here
    time1 = time.time()
    torch_solver.run_pf(report_metrics=True)
    time2 = time.time()
    print("time: ", time2 - time1)  # 0.586 for 200 iterations
    loss_list = torch_solver.loss_list
    best_loss = torch_solver.best_loss
    print("best loss: ", best_loss)

    metrics = torch_solver.eval_dict
    mse = metrics["mse"]
    average_percentage_diff = torch_solver.eval_dict["average_percentage_diff"]

    """
    # targets
    target_V_solver = torch_solver.V
    target_Va_solver = torch_solver.Va
    target_Vm_solver = torch_solver.Vm

    torch_solver.post_process()
    theta_or_calc, theta_ex_calc, v_or_calc, v_ex_calc, a_or_calc, a_ex_calc, p_or_calc, p_ex_calc, q_or_calc, q_ex_calc = (
        torch_solver.extract_results())
    theta_or_diff = (abs(theta_or_calc - theta_or) >= 0.01).sum()
    theta_ex_diff = (abs(theta_ex_calc - theta_ex) >= 0.01).sum()
    v_or_diff = (abs(v_or_calc - v_or) >= 0.01).sum()
    v_ex_diff = (abs(v_ex_calc - v_ex) >= 0.01).sum()
    p_or_diff = (abs(p_or_calc - p_or) >= 0.01).sum()
    p_ex_diff = (abs(p_ex_calc - p_ex) >= 0.01).sum()
    a_or_diff = (abs(a_or_calc - a_or) >= 0.01).sum()
    a_ex_diff = (abs(a_ex_calc - a_ex) >= 0.01).sum()
    print("differences: ", theta_or_diff, theta_ex_diff, v_or_diff, v_ex_diff, p_or_diff, p_ex_diff, a_or_diff,
          a_ex_diff)

     print(loss_list)
     print(" ")
    """

    # show mse loss here
    # plt.plot(loss_list)
    # plt.plot(mse)
    plt.plot(average_percentage_diff * 100)
    plt.title("Training curve")
    plt.xlabel("Iterations")
    plt.ylabel("Mean flow difference in %")
    plt.savefig("out/plots/ex1_average_percentage_diff.png")


if __name__ == "__main__":
    main()

# run with: python src/dpf/scripts/ex1_running_torch_solver.py
# or via pyproject.toml scripts via run-ex1 after installing

#
#  Goal: Given specific loss (% mismatch) find # iterations to achieve that loss

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


def run_experiment(
    strategy="NR",
    sample=2,
    to_optimizer="Adam",
    case_name="case2848pegase",
    use_gpu=False,
    save_result=False,
):
    # load data
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
    num_repetitions = 5

    if strategy == "NR":
        # output: list of times/voltages/mismatches each iteration
        # output: single time and metrics on power mismatch
        times_list = []
        average_percentage_diffes_list = []
        losses_list = []
        for _ in range(num_repetitions):
            solver = NRPowerFlowSolverCPP(backend=backend)
            solver.preprocess(
                topo_vect, prod_p, prod_v, load_p, load_q, Ybus, Sbus, PV_nodes
            )
            start_time = time.perf_counter()
            solver.init_v("dc")
            solver.run_pf()
            end_time = time.perf_counter()
            print("NR time: ", end_time - start_time)

            times = []
            times.append(end_time - start_time)
            average_percentage_diffes_list.append(
                solver.calculate_average_percentage_diff()
            )
            losses_list.append(solver.calculate_l2_loss())
            times_list.append(times)

        times_list = numpy.array(times_list)
        mean_time = times_list.mean()

        average_percentage_diffes_list = numpy.array(average_percentage_diffes_list)
        mean_average_percentage_diff = average_percentage_diffes_list.mean()

        losses_list = numpy.array(losses_list)
        mean_loss = losses_list.mean()

        results = {
            "times": mean_time,
            "avg_percentage_diff": mean_average_percentage_diff,
            "losses": mean_loss,
        }
        if save_result:
            with open(
                f"out/temp/ex10a_NR_{case_name}_{use_gpu}.pkl", "wb"
            ) as writeNRFile:
                pickle.dump(results, writeNRFile)

    if strategy == "TO":
        # output: list of times/voltages/mismatches each iteration
        # generic hyperaprameters

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

        if case_name == "case118":
            hyperparams = {
                "optimizer_class": torch.optim.Adam,
                "optimizer_kwargs": {
                    "lr": 0.01795038617625319,
                    "betas": (0.9763712103234077, 0.9586488750093097),
                },
                "scheduler_class": torch.optim.lr_scheduler.ReduceLROnPlateau,
                "scheduler_kwargs": {
                    "factor": 0.8909046972623936,
                    "patience": 14,
                    "threshold_mode": "rel",
                    "threshold": 0.03789578966995357,
                    "cooldown": 18,
                },
                "loss_fn": torch.nn.MSELoss(),
                "max_iter": 1000,
                "tol": 1e-8,
            }
            hyperparams = {
                "optimizer_class": torch.optim.Adam,
                "optimizer_kwargs": {
                    "lr": 0.026506043268263228,
                    "betas": (0.9133120862183772, 0.9191237715987598),
                },
                "scheduler_class": torch.optim.lr_scheduler.ReduceLROnPlateau,
                "scheduler_kwargs": {
                    "factor": 0.5423890808281354,
                    "patience": 91,
                    "threshold_mode": "rel",
                    "threshold": 0.005851025226150188,
                    "cooldown": 90,
                },
                "loss_fn": torch.nn.MSELoss(),
                "max_iter": 10000,
                "tol": 1e-8,
            }
            hyperparams = {
                "optimizer_class": torch.optim.Adam,
                "optimizer_kwargs": {
                    "lr": 0.0038228432728267107,
                    "betas": (0.9978031774325471, 0.9972913604183515),
                },
                "scheduler_class": torch.optim.lr_scheduler.ConstantLR,
                "scheduler_kwargs": {"factor": 1.0, "total_iters": 9999999999},
                "loss_fn": torch.nn.MSELoss(),
                "max_iter": 10000,
                "tol": 1e-8,
            }
        if case_name == "case_illinois200":
            hyperparams = {
                "optimizer_class": torch.optim.Adam,
                "optimizer_kwargs": {
                    "lr": 0.012289517793645479,
                    "betas": (0.976837705179429, 0.9801263819402884),
                },
                "scheduler_class": torch.optim.lr_scheduler.ReduceLROnPlateau,
                "scheduler_kwargs": {
                    "factor": 0.7319650773879361,
                    "patience": 40,
                    "threshold_mode": "rel",
                    "threshold": 0.0579828155695889,
                    "cooldown": 18,
                },
                "loss_fn": torch.nn.MSELoss(),
                "max_iter": 1000,
                "tol": 1e-8,
            }
            hyperparams = {
                "optimizer_class": torch.optim.Adam,
                "optimizer_kwargs": {
                    "lr": 0.00845802168747944,
                    "betas": (0.984060746602013, 0.9811416052536659),
                },
                "scheduler_class": torch.optim.lr_scheduler.ReduceLROnPlateau,
                "scheduler_kwargs": {
                    "factor": 0.9633766597887508,
                    "patience": 54,
                    "threshold_mode": "rel",
                    "threshold": 0.08167130876471938,
                    "cooldown": 7,
                },
                "loss_fn": torch.nn.MSELoss(),
                "max_iter": 10000,
                "tol": 1e-8,
            }
            hyperparams = {
                "optimizer_class": torch.optim.Adam,
                "optimizer_kwargs": {
                    "lr": 0.001108469877949029,
                    "betas": (0.9813897477018912, 0.9871241111646865),
                },
                "scheduler_class": torch.optim.lr_scheduler.ConstantLR,
                "scheduler_kwargs": {"factor": 1.0, "total_iters": 9999999999},
                "loss_fn": torch.nn.MSELoss(),
                "max_iter": 10000,
                "tol": 1e-8,
            }
        if case_name == "case300":
            hyperparams = {
                "optimizer_class": torch.optim.Adam,
                "optimizer_kwargs": {
                    "lr": 0.009521723757059015,
                    "betas": (0.9712532256241938, 0.9463427160772577),
                },
                "scheduler_class": torch.optim.lr_scheduler.ReduceLROnPlateau,
                "scheduler_kwargs": {
                    "factor": 0.4845831316546101,
                    "patience": 36,
                    "threshold_mode": "rel",
                    "threshold": 0.0696814334360944,
                    "cooldown": 28,
                },
                "loss_fn": torch.nn.MSELoss(),
                "max_iter": 1000,
                "tol": 1e-8,
            }
            hyperparams = {
                "optimizer_class": torch.optim.Adam,
                "optimizer_kwargs": {
                    "lr": 0.007260824938742863,
                    "betas": (0.9796279980503929, 0.9476686918784878),
                },
                "scheduler_class": torch.optim.lr_scheduler.ReduceLROnPlateau,
                "scheduler_kwargs": {
                    "factor": 0.923253359752288,
                    "patience": 78,
                    "threshold_mode": "rel",
                    "threshold": 0.06785828740373034,
                    "cooldown": 95,
                },
                "loss_fn": torch.nn.MSELoss(),
                "max_iter": 10000,
                "tol": 1e-8,
            }
            hyperparams = {
                "optimizer_class": torch.optim.Adam,
                "optimizer_kwargs": {
                    "lr": 0.0017053201783004818,
                    "betas": (0.9859229997118575, 0.9884910997706813),
                },
                "scheduler_class": torch.optim.lr_scheduler.ConstantLR,
                "scheduler_kwargs": {"factor": 1.0, "total_iters": 9999999999},
                "loss_fn": torch.nn.MSELoss(),
                "max_iter": 10000,
                "tol": 1e-8,
            }
        if case_name == "case1354pegase":
            hyperparams = {
                "optimizer_class": torch.optim.Adam,
                "optimizer_kwargs": {
                    "lr": 0.004639998357422333,
                    "betas": (0.9656571694057735, 0.9326783792460925),
                },
                "scheduler_class": torch.optim.lr_scheduler.ReduceLROnPlateau,
                "scheduler_kwargs": {
                    "factor": 0.12709779638884938,
                    "patience": 44,
                    "threshold_mode": "rel",
                    "threshold": 0.01112233080171757,
                    "cooldown": 61,
                },
                "loss_fn": torch.nn.MSELoss(),
                "max_iter": 1000,
                "tol": 1e-8,
            }
            hyperparams = {
                "optimizer_class": torch.optim.Adam,
                "optimizer_kwargs": {
                    "lr": 0.012536441014909243,
                    "betas": (0.9880519630303984, 0.9817149707473227),
                },
                "scheduler_class": torch.optim.lr_scheduler.ReduceLROnPlateau,
                "scheduler_kwargs": {
                    "factor": 0.8858786652069653,
                    "patience": 63,
                    "threshold_mode": "rel",
                    "threshold": 0.013916229613337728,
                    "cooldown": 12,
                },
                "loss_fn": torch.nn.MSELoss(),
                "max_iter": 10000,
                "tol": 1e-8,
            }
            hyperparams = {
                "optimizer_class": torch.optim.Adam,
                "optimizer_kwargs": {
                    "lr": 0.0002582755140634165,
                    "betas": (0.991285030278202, 0.9710029344021375),
                },
                "scheduler_class": torch.optim.lr_scheduler.ConstantLR,
                "scheduler_kwargs": {"factor": 1.0, "total_iters": 9999999999},
                "loss_fn": torch.nn.MSELoss(),
                "max_iter": 10000,
                "tol": 1e-8,
            }
        if case_name == "case1888rte":
            hyperparams = {
                "optimizer_class": torch.optim.Adam,
                "optimizer_kwargs": {
                    "lr": 0.0033512648122632754,
                    "betas": (0.950558499287691, 0.9144993259604787),
                },
                "scheduler_class": torch.optim.lr_scheduler.ReduceLROnPlateau,
                "scheduler_kwargs": {
                    "factor": 0.593488228237255,
                    "patience": 61,
                    "threshold_mode": "rel",
                    "threshold": 0.08578661575373467,
                    "cooldown": 29,
                },
                "loss_fn": torch.nn.MSELoss(),
                "max_iter": 1000,
                "tol": 1e-8,
            }
            hyperparams = {
                "optimizer_class": torch.optim.Adam,
                "optimizer_kwargs": {
                    "lr": 0.0027543046833169017,
                    "betas": (0.9911160723239226, 0.9936639121472417),
                },
                "scheduler_class": torch.optim.lr_scheduler.ReduceLROnPlateau,
                "scheduler_kwargs": {
                    "factor": 0.707330985207001,
                    "patience": 67,
                    "threshold_mode": "rel",
                    "threshold": 0.0004067789676258683,
                    "cooldown": 43,
                },
                "loss_fn": torch.nn.MSELoss(),
                "max_iter": 10000,
                "tol": 1e-8,
            }
            hyperparams = {
                "optimizer_class": torch.optim.Adam,
                "optimizer_kwargs": {
                    "lr": 0.0021783176993847393,
                    "betas": (0.973040825662213, 0.9976111701218154),
                },
                "scheduler_class": torch.optim.lr_scheduler.ConstantLR,
                "scheduler_kwargs": {"factor": 1.0, "total_iters": 9999999999},
                "loss_fn": torch.nn.MSELoss(),
                "max_iter": 10000,
                "tol": 1e-8,
            }
        if case_name == "case2869pegase":
            hyperparams = {
                "optimizer_class": torch.optim.Adam,
                "optimizer_kwargs": {
                    "lr": 0.0026822631108718575,
                    "betas": (0.9635765173710168, 0.9403072711831784),
                },
                "scheduler_class": torch.optim.lr_scheduler.ReduceLROnPlateau,
                "scheduler_kwargs": {
                    "factor": 0.5952165742500408,
                    "patience": 45,
                    "threshold_mode": "rel",
                    "threshold": 0.0848390212523344,
                    "cooldown": 50,
                },
                "loss_fn": torch.nn.MSELoss(),
                "max_iter": 1000,
                "tol": 1e-8,
            }
            hyperparams = {
                "optimizer_class": torch.optim.Adam,
                "optimizer_kwargs": {
                    "lr": 0.001700588383074072,
                    "betas": (0.9953085258676201, 0.9870948213029219),
                },
                "scheduler_class": torch.optim.lr_scheduler.ReduceLROnPlateau,
                "scheduler_kwargs": {
                    "factor": 0.9678206212001057,
                    "patience": 37,
                    "threshold_mode": "rel",
                    "threshold": 0.0842572715914738,
                    "cooldown": 30,
                },
                "loss_fn": torch.nn.MSELoss(),
                "max_iter": 10000,
                "tol": 1e-8,
            }
            hyperparams = {
                "optimizer_class": torch.optim.Adam,
                "optimizer_kwargs": {
                    "lr": 0.0012999605801802424,
                    "betas": (0.9977113173287585, 0.9964950593853412),
                },
                "scheduler_class": torch.optim.lr_scheduler.ConstantLR,
                "scheduler_kwargs": {"factor": 1.0, "total_iters": 9999999999},
                "loss_fn": torch.nn.MSELoss(),
                "max_iter": 10000,
                "tol": 1e-8,
            }
        if case_name == "case3120sp":
            hyperparams = {
                "optimizer_class": torch.optim.Adam,
                "optimizer_kwargs": {
                    "lr": 0.002088058570251777,
                    "betas": (0.7496513026093589, 0.8055282769660045),
                },
                "scheduler_class": torch.optim.lr_scheduler.ReduceLROnPlateau,
                "scheduler_kwargs": {
                    "factor": 0.45616793242590403,
                    "patience": 70,
                    "threshold_mode": "rel",
                    "threshold": 0.02260296280805881,
                    "cooldown": 32,
                },
                "loss_fn": torch.nn.MSELoss(),
                "max_iter": 1000,
                "tol": 1e-8,
            }
            hyperparams = {
                "optimizer_class": torch.optim.Adam,
                "optimizer_kwargs": {
                    "lr": 0.003532710193263489,
                    "betas": (0.9847093477072952, 0.9897042913025187),
                },
                "scheduler_class": torch.optim.lr_scheduler.ReduceLROnPlateau,
                "scheduler_kwargs": {
                    "factor": 0.8422705826455478,
                    "patience": 72,
                    "threshold_mode": "rel",
                    "threshold": 0.013840233145982012,
                    "cooldown": 56,
                },
                "loss_fn": torch.nn.MSELoss(),
                "max_iter": 10000,
                "tol": 1e-8,
            }
            hyperparams = {
                "optimizer_class": torch.optim.Adam,
                "optimizer_kwargs": {
                    "lr": 0.00021966986414244177,
                    "betas": (0.9618768877389932, 0.950058716940527),
                },
                "scheduler_class": torch.optim.lr_scheduler.ConstantLR,
                "scheduler_kwargs": {"factor": 1.0, "total_iters": 9999999999},
                "loss_fn": torch.nn.MSELoss(),
                "max_iter": 10000,
                "tol": 1e-8,
            }
        if case_name == "case6495rte":
            hyperparams = {
                "optimizer_class": torch.optim.Adam,
                "optimizer_kwargs": {
                    "lr": 0.005657171451727259,
                    "betas": (0.9756220765879436, 0.9723036580101386),
                },
                "scheduler_class": torch.optim.lr_scheduler.ReduceLROnPlateau,
                "scheduler_kwargs": {
                    "factor": 0.5226567065237384,
                    "patience": 69,
                    "threshold_mode": "rel",
                    "threshold": 0.09012166735533989,
                    "cooldown": 44,
                },
                "loss_fn": torch.nn.MSELoss(),
                "max_iter": 1000,
                "tol": 1e-8,
            }
            hyperparams = {
                "optimizer_class": torch.optim.Adam,
                "optimizer_kwargs": {
                    "lr": 0.016651396490155155,
                    "betas": (0.9790280941156596, 0.9836402036928811),
                },
                "scheduler_class": torch.optim.lr_scheduler.ReduceLROnPlateau,
                "scheduler_kwargs": {
                    "factor": 0.7488405702351403,
                    "patience": 30,
                    "threshold_mode": "rel",
                    "threshold": 0.00025996245636666394,
                    "cooldown": 20,
                },
                "loss_fn": torch.nn.MSELoss(),
                "max_iter": 10000,
                "tol": 1e-8,
            }
            hyperparams = {
                "optimizer_class": torch.optim.Adam,
                "optimizer_kwargs": {
                    "lr": 0.00019308563166126522,
                    "betas": (0.9799878664944089, 0.9311001907415796),
                },
                "scheduler_class": torch.optim.lr_scheduler.ConstantLR,
                "scheduler_kwargs": {"factor": 1.0, "total_iters": 9999999999},
                "loss_fn": torch.nn.MSELoss(),
                "max_iter": 10000,
                "tol": 1e-8,
            }
        if case_name == "case6515rte":
            hyperparams = {
                "optimizer_class": torch.optim.Adam,
                "optimizer_kwargs": {
                    "lr": 0.011778085150342534,
                    "betas": (0.9616132151627467, 0.9567943906116894),
                },
                "scheduler_class": torch.optim.lr_scheduler.ReduceLROnPlateau,
                "scheduler_kwargs": {
                    "factor": 0.318124767620211,
                    "patience": 26,
                    "threshold_mode": "rel",
                    "threshold": 0.010512007585684033,
                    "cooldown": 95,
                },
                "loss_fn": torch.nn.MSELoss(),
                "max_iter": 1000,
                "tol": 1e-8,
            }
            hyperparams = {
                "optimizer_class": torch.optim.Adam,
                "optimizer_kwargs": {
                    "lr": 0.003996105385449172,
                    "betas": (0.9909339187507291, 0.9776194935600365),
                },
                "scheduler_class": torch.optim.lr_scheduler.ReduceLROnPlateau,
                "scheduler_kwargs": {
                    "factor": 0.3755226713655897,
                    "patience": 78,
                    "threshold_mode": "rel",
                    "threshold": 0.0100099304529931,
                    "cooldown": 62,
                },
                "loss_fn": torch.nn.MSELoss(),
                "max_iter": 10000,
                "tol": 1e-8,
            }
            hyperparams = {
                "optimizer_class": torch.optim.Adam,
                "optimizer_kwargs": {
                    "lr": 0.002596890676885278,
                    "betas": (0.9923468858674624, 0.9868064282154224),
                },
                "scheduler_class": torch.optim.lr_scheduler.ConstantLR,
                "scheduler_kwargs": {"factor": 1.0, "total_iters": 9999999999},
                "loss_fn": torch.nn.MSELoss(),
                "max_iter": 10000,
                "tol": 1e-8,
            }

        if case_name == "case9241pegase":
            hyperparams = {
                "optimizer_class": torch.optim.Adam,
                "optimizer_kwargs": {
                    "lr": 0.004686149671315877,
                    "betas": (0.9569887851973654, 0.9429589128128013),
                },
                "scheduler_class": torch.optim.lr_scheduler.ReduceLROnPlateau,
                "scheduler_kwargs": {
                    "factor": 0.635347737607972,
                    "patience": 68,
                    "threshold_mode": "rel",
                    "threshold": 0.06005638814082652,
                    "cooldown": 29,
                },
                "loss_fn": torch.nn.MSELoss(),
                "max_iter": 1000,
                "tol": 1e-8,
            }
            hyperparams = {
                "optimizer_class": torch.optim.Adam,
                "optimizer_kwargs": {
                    "lr": 0.009910535586096234,
                    "betas": (0.9906999148323791, 0.9827309992216938),
                },
                "scheduler_class": torch.optim.lr_scheduler.ReduceLROnPlateau,
                "scheduler_kwargs": {
                    "factor": 0.9748242293910261,
                    "patience": 20,
                    "threshold_mode": "rel",
                    "threshold": 0.04752888073711975,
                    "cooldown": 66,
                },
                "loss_fn": torch.nn.MSELoss(),
                "max_iter": 10000,
                "tol": 1e-8,
            }
            hyperparams = {
                "optimizer_class": torch.optim.Adam,
                "optimizer_kwargs": {
                    "lr": 0.00590263467583827,
                    "betas": (0.9947645411289673, 0.9913162240085669),
                },
                "scheduler_class": torch.optim.lr_scheduler.ConstantLR,
                "scheduler_kwargs": {"factor": 1.0, "total_iters": 9999999999},
                "loss_fn": torch.nn.MSELoss(),
                "max_iter": 10000,
                "tol": 1e-8,
            }

        times_list = []
        loss_list = []
        avg_percentage_diff_list = []

        for _ in range(num_repetitions):
            solver = TorchPowerFlowSolver(backend=backend, hyperparams=hyperparams)

            solver.set_gpu_usage(use_gpu=use_gpu)
            device = solver.device

            solver.preprocess(
                topo_vect, prod_p, prod_v, load_p, load_q, Ybus, Sbus, PV_nodes
            )
            start_time = time.perf_counter()
            solver.init_v("ones")
            time_before_pf = time.perf_counter()
            solver.run_pf(checkpointing=False, report_metrics=True)
            end_time = time.perf_counter()

            times = [x + (time_before_pf - start_time) for x in solver.times_list]
            mismatches_mse = list(solver.loss_list)

            metrics = solver.eval_dict
            average_percentage_diff_list = metrics["average_percentage_diff"]

            mismatches_mse.append(solver.calculate_l2_loss())
            times.append(end_time - start_time)

            times_list.append(times)
            loss_list.append(mismatches_mse)
            avg_percentage_diff_list.append(average_percentage_diff_list)
            # print(average_percentage_diff_list)

        times_list = numpy.array(times_list)
        mean_times = [sum(values) / len(values) for values in zip(*times_list)]
        mean_times = list(mean_times)

        avg_percentage_diff_list = numpy.array(avg_percentage_diff_list)
        mean_average_percentage_diffes = [
            sum(values) / len(values) for values in zip(*avg_percentage_diff_list)
        ]
        mean_average_percentage_diffes = list(mean_average_percentage_diffes)

        loss_list = numpy.array(loss_list)
        mean_losses = [sum(values) / len(values) for values in zip(*loss_list)]
        mean_losses = list(mean_losses)

        results = {
            "times": mean_times,
            "avg_percentage_diff": mean_average_percentage_diffes,
            "loss": mean_losses,
        }

        if save_result:
            with open(
                f"out/temp/ex10a_TO_{to_optimizer}_{case_name}_{use_gpu}.pkl", "wb"
            ) as writeTOFile:
                pickle.dump(results, writeTOFile)


def main():
    import warnings

    warnings.filterwarnings("ignore")

    print("Starting experiment 10a")
    sample = 2
    use_gpu = False

    case_names = [
        "case118",
        "case_illinois200",
        "case300",
        "case1354pegase",
        "case1888rte",
        "case2869pegase",
        "case3120sp",
        "case6495rte",
        "case6515rte",
        "case9241pegase",
    ]
    grid_sizes = [118, 200, 300, 1354, 1888, 2869, 3120, 6495, 6515, 9241]

    to_optimizer = "Adam"

    methods = ["NR", "TO"]

    for method in methods:
        run_experiment(
            "DC", sample, "None", "case118", use_gpu=use_gpu, save_result=False
        )  # warm start
        for case_name in case_names:
            run_experiment(
                method,
                sample,
                to_optimizer,
                case_name,
                use_gpu=use_gpu,
                save_result=True,
            )


if __name__ == "__main__":
    main()

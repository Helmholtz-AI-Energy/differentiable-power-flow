import numpy as np
import optuna
import torch.optim
import os
import pandapower as pp

from dpf.dataset import CustomGridDataset
from dpf.scripts.ex4_data_generation import make_grid2op_env, get_loads_gens
from dpf.solvers.solver_torch import TorchPowerFlowSolver


def objective(trial, inputs, targets, env, backend, fixed_optimizer_name=None, fixed_scheduler_strat=None,
              max_iter_fixed=None, fixed_loss_fn=None):
    if fixed_optimizer_name is not None:
        optimizer_name = fixed_optimizer_name
    else:
        optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "RMSprop", "SGD"])
    if fixed_scheduler_strat is not None:
        scheduler_strat = fixed_scheduler_strat
    else:
        scheduler_strat = trial.suggest_categorical("scheduler", ["constant", "StepLR", "ReduceLROnPlateau"])
    if max_iter_fixed is not None:
        max_iter = max_iter_fixed
    else:
        max_iter = 1000
    if fixed_loss_fn is not None:
        loss_fn_name = fixed_loss_fn
    else:
        loss_fn_name = trial.suggest_categorical("loss_fn", ["L1", "L1Sum", "MSE", "Huber"])

    prod_p, prod_v, load_p, load_q, line_status, topo_vect, Ybus, Sbus, PV_nodes, slack = inputs
    a_or, a_ex, p_or, p_ex, v_or, v_ex, theta_or, theta_ex = targets  # maybe use these for evaluation as well?

    optimizer_class = None
    optimizer_kwargs = None
    if optimizer_name == "Adam":
        lr = trial.suggest_float("lr", 1e-6, 1e-1)  # learning rate
        beta_one = trial.suggest_float("beta1", .5, 0.999)  # exponential decay rate for momentum
        beta_two = trial.suggest_float("beta2", 0.8, 0.9999999)  # exponential decay rate for velocity
        optimizer_class = torch.optim.Adam
        optimizer_kwargs = {"lr": lr, "betas": (beta_one, beta_two)}
    if optimizer_name == "SGD":
        lr = trial.suggest_float("lr", 1e-6, 1e-1)
        momentum = trial.suggest_float("momentum", 0.0, 0.99)
        dampening = trial.suggest_float("dampening", 0.0, 0.99)
        weight_decay = trial.suggest_float("weight_decay", 0.0, 0.99)
        optimizer_class = torch.optim.SGD
        optimizer_kwargs = {"lr": lr, "momentum": momentum, "dampening": dampening, "weight_decay": weight_decay}
    if optimizer_name == "RMSprop":
        lr = trial.suggest_float("lr", 1e-6, 1e-1)
        alpha = trial.suggest_float("alpha", 0.1, 0.999)
        weight_decay = trial.suggest_float("weight_decay", 0.0, 0.999)
        momentum = trial.suggest_float("momentum", 0.0, 0.999)
        optimizer_class = torch.optim.RMSprop
        optimizer_kwargs = {"lr": lr, "alpha": alpha, "weight_decay": weight_decay, "momentum": momentum}

    scheduler_class = None
    scheduler_kwargs = None
    if scheduler_strat == "constant":
        scheduler_class = torch.optim.lr_scheduler.ConstantLR
        scheduler_kwargs = {"factor": 1.0, "total_iters": 9999999999}
    if scheduler_strat == "StepLR":
        step_size = trial.suggest_int("step_size", 1, 100)
        gamma = trial.suggest_float("gamma", 0.0001, 0.99)
        scheduler_class = torch.optim.lr_scheduler.StepLR
        scheduler_kwargs = {"step_size": step_size, "gamma": gamma}
    if scheduler_strat == "ReduceLROnPlateau":
        factor = trial.suggest_float("factor", 0.01, 0.99)
        patience = trial.suggest_int("patience", 0, 100)
        # threshold_mode = trial.suggest_categorical("threshold", ["rel", "abs"])
        threshold_mode = "rel"
        threshold = trial.suggest_float("threshold", 1e-6, 1e-1)
        cooldown = trial.suggest_int("cooldown", 0, 100)
        scheduler_class = torch.optim.lr_scheduler.ReduceLROnPlateau
        scheduler_kwargs = {"factor": factor, "patience": patience, "threshold_mode": threshold_mode,
                            "threshold": threshold, "cooldown": cooldown}
    if scheduler_strat == "MultiStepLR":
        num_milestones = trial.suggest_int("num_milestones", 1, 20)
        milestones = sorted(trial.suggest_int("milestone_" + str(i), 1, 300) for i in range(num_milestones))
        gamma = trial.suggest_float("gamma", 0.0001, 0.99)
        scheduler_class = torch.optim.lr_scheduler.MultiStepLR
        scheduler_kwargs = {"milestones": milestones, "gamma": gamma}

    loss_fn = None

    if loss_fn_name == "L1":  # ["L1", "MSE", "CE"])
        loss_fn = torch.nn.L1Loss()
    if loss_fn_name == "L1Sum":
        loss_fn = torch.nn.L1Loss(reduction="sum")
    if loss_fn_name == "MSE":
        loss_fn = torch.nn.MSELoss()
    if loss_fn_name == "CE":
        loss_fn = torch.nn.CrossEntropyLoss()  # this somehow does not work
    if loss_fn_name == "Huber":
        loss_fn = torch.nn.HuberLoss()

    hyperparams = {
        "optimizer_class": optimizer_class,
        "optimizer_kwargs": optimizer_kwargs,
        "scheduler_class": scheduler_class,
        "scheduler_kwargs": scheduler_kwargs,
        #"loss_fn": torch.nn.MSELoss(),
        "loss_fn": loss_fn,
        "max_iter": max_iter,
        "tol": 1e-8
    }

    # run experiment:

    solver = TorchPowerFlowSolver(backend=backend, hyperparams=hyperparams)
    solver.preprocess(topo_vect, prod_p, prod_v, load_p, load_q, Ybus, Sbus, PV_nodes)
    solver.init_v("ones")
    solver.run_pf(report_metrics=True)

    loss = solver.eval_dict["mse"][-1]

    return loss


def run_experiment(case_name, optimizer_strat, scheduler_strat, max_iter, loss_fun, n_trials):
    sample = 2
    # case_name = "case_illinois200"
    custom_grid_dataset = CustomGridDataset(env_name=case_name)
    inputs, targets = custom_grid_dataset.get_sample(sample)
    prod_p, prod_v, load_p, load_q, line_status, topo_vect, Ybus, Sbus, PV_nodes, slack = inputs
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

    study = optuna.create_study()
    study.optimize(
        lambda trial: objective(trial, inputs, targets, env, backend, optimizer_strat, scheduler_strat, max_iter,
                                loss_fun),
        n_trials=n_trials)
    print(f"Best params is {study.best_params} with value {study.best_value}")

    df = study.trials_dataframe(attrs=("params", "value", "state"))
    df.to_csv(f"out/temp/ex10_trials_{case_name}_{optimizer_strat}_{scheduler_strat}.csv",
              index=False)


def main():
    n_trials = 100

    # go through all combinations
    # optimizer_strat = "Adam"
    #scheduler_strat = "ReduceLROnPlateau"

    optimizer_strat = "Adam"
    scheduler_strat = "constant"
    num_iter = 10000
    loss_fn = "MSE"

    case_names = ["case118", "case_illinois200", "case300", "case1354pegase", "case1888rte",
                  "case2869pegase", "case3120sp", "case6495rte", "case6515rte", "case9241pegase"]
    #case_names = ["case1888rte", "case2869pegase", "case3120sp", "case6495rte", "case6515rte", "case9241pegase"]

    for case_name in case_names:
        run_experiment(case_name, optimizer_strat, scheduler_strat, num_iter, loss_fun=loss_fn, n_trials=n_trials, )

    ##  for 1000 iterations

    # Best params is {'lr': 0.01795038617625319, 'beta1': 0.9763712103234077, 'beta2': 0.9586488750093097, 'factor': 0.8909046972623936, 'patience': 14, 'threshold': 0.03789578966995357, 'cooldown': 18} with value 0.0004806076920065489
    # Best params is {'lr': 0.012289517793645479, 'beta1': 0.976837705179429, 'beta2': 0.9801263819402884, 'factor': 0.7319650773879361, 'patience': 40, 'threshold': 0.0579828155695889, 'cooldown': 18} with value 0.0003357665574519552
    # Best params is {'lr': 0.009521723757059015, 'beta1': 0.9712532256241938, 'beta2': 0.9463427160772577, 'factor': 0.4845831316546101, 'patience': 36, 'threshold': 0.0696814334360944, 'cooldown': 28} with value 0.001940475236122593
    # Best params is {'lr': 0.004639998357422333, 'beta1': 0.9656571694057735, 'beta2': 0.9326783792460925, 'factor': 0.12709779638884938, 'patience': 44, 'threshold': 0.01112233080171757, 'cooldown': 61} with value 0.09309180277039024
    # Best params is {'lr': 0.0033512648122632754, 'beta1': 0.950558499287691, 'beta2': 0.9144993259604787, 'factor': 0.593488228237255, 'patience': 61, 'threshold': 0.08578661575373467, 'cooldown': 29} with value 0.15162112668812064
    # Best params is {'lr': 0.0026822631108718575, 'beta1': 0.9635765173710168, 'beta2': 0.9403072711831784, 'factor': 0.5952165742500408, 'patience': 45, 'threshold': 0.0848390212523344, 'cooldown': 50} with value 0.0821612519398687
    # Best params is {'lr': 0.002088058570251777, 'beta1': 0.7496513026093589, 'beta2': 0.8055282769660045, 'factor': 0.45616793242590403, 'patience': 70, 'threshold': 0.02260296280805881, 'cooldown': 32} with value 0.03971005429280965
    # Best params is {'lr': 0.005657171451727259, 'beta1': 0.9756220765879436, 'beta2': 0.9723036580101386, 'factor': 0.5226567065237384, 'patience': 69, 'threshold': 0.09012166735533989, 'cooldown': 44} with value 0.042523683420508634
    # Best params is {'lr': 0.011778085150342534, 'beta1': 0.9616132151627467, 'beta2': 0.9567943906116894, 'factor': 0.318124767620211, 'patience': 26, 'threshold': 0.010512007585684033, 'cooldown': 95} with value 0.051938893858982385
    # Best params is {'lr': 0.004686149671315877, 'beta1': 0.9569887851973654, 'beta2': 0.9429589128128013, 'factor': 0.635347737607972, 'patience': 68, 'threshold': 0.06005638814082652, 'cooldown': 29} with value 0.0527880869978418

    # for 10000 iterations

    # {'lr': 0.026506043268263228, 'beta1': 0.9133120862183772, 'beta2': 0.9191237715987598, 'factor': 0.5423890808281354, 'patience': 91, 'threshold': 0.005851025226150188, 'cooldown': 90} with value 0.00017531354624708716
    # {'lr': 0.00845802168747944, 'beta1': 0.984060746602013, 'beta2': 0.9811416052536659, 'factor': 0.9633766597887508, 'patience': 54, 'threshold': 0.08167130876471938, 'cooldown': 7} with value 0.00011061263922035771
    # {'lr': 0.007260824938742863, 'beta1': 0.9796279980503929, 'beta2': 0.9476686918784878, 'factor': 0.923253359752288, 'patience': 78, 'threshold': 0.06785828740373034, 'cooldown': 95} with value 0.00038090153638045297
    # {'lr': 0.012536441014909243, 'beta1': 0.9880519630303984, 'beta2': 0.9817149707473227, 'factor': 0.8858786652069653, 'patience': 63, 'threshold': 0.013916229613337728, 'cooldown': 12} with value 0.006616265345903823
    # {'lr': 0.0027543046833169017, 'beta1': 0.9911160723239226, 'beta2': 0.9936639121472417, 'factor': 0.707330985207001, 'patience': 67, 'threshold': 0.0004067789676258683, 'cooldown': 43} with value 0.015694339673405614
    # {'lr': 0.001700588383074072, 'beta1': 0.9953085258676201, 'beta2': 0.9870948213029219, 'factor': 0.9678206212001057, 'patience': 37, 'threshold': 0.0842572715914738, 'cooldown': 30} with value 0.0040030344251424255
    # {'lr': 0.003532710193263489, 'beta1': 0.9847093477072952, 'beta2': 0.9897042913025187, 'factor': 0.8422705826455478, 'patience': 72, 'threshold': 0.013840233145982012, 'cooldown': 56} with value 0.002140871283931944
    # {'lr': 0.016651396490155155, 'beta1': 0.9790280941156596, 'beta2': 0.9836402036928811, 'factor': 0.7488405702351403, 'patience': 30, 'threshold': 0.00025996245636666394, 'cooldown': 20} with value 0.007755677475538409
    # {'lr': 0.003996105385449172, 'beta1': 0.9909339187507291, 'beta2': 0.9776194935600365, 'factor': 0.3755226713655897, 'patience': 78, 'threshold': 0.0100099304529931, 'cooldown': 62} with value 0.006140599752484576
    # {'lr': 0.009910535586096234, 'beta1': 0.9906999148323791, 'beta2': 0.9827309992216938, 'factor': 0.9748242293910261, 'patience': 20, 'threshold': 0.04752888073711975, 'cooldown': 66} with value 0.0074542188461988795

    # for Adam + constant scheduler with 10k iterations
    # Best params is {'lr': 0.0038228432728267107, 'beta1': 0.9978031774325471, 'beta2': 0.9972913604183515} with value 1.5488776251972474e-07
    # Best params is {'lr': 0.001108469877949029, 'beta1': 0.9813897477018912, 'beta2': 0.9871241111646865} with value 0.000122752478201701
    # Best params is {'lr': 0.0017053201783004818, 'beta1': 0.9859229997118575, 'beta2': 0.9884910997706813} with value 0.0002762985784986401
    # Best params is {'lr': 0.0002582755140634165, 'beta1': 0.991285030278202, 'beta2': 0.9710029344021375} with value 0.005745124342748584
    # Best params is {'lr': 0.0021783176993847393, 'beta1': 0.973040825662213, 'beta2': 0.9976111701218154} with value 0.041944139141340285
    # Best params is {'lr': 0.0012999605801802424, 'beta1': 0.9977113173287585, 'beta2': 0.9964950593853412} with value 0.002544821999385515
    # Best params is {'lr': 0.00021966986414244177, 'beta1': 0.9618768877389932, 'beta2': 0.950058716940527} with value 0.005122509513808729
    # Best params is {'lr': 0.00019308563166126522, 'beta1': 0.9799878664944089, 'beta2': 0.9311001907415796} with value 0.009682243328429083
    # Best params is {'lr': 0.002596890676885278, 'beta1': 0.9923468858674624, 'beta2': 0.9868064282154224} with value 0.004828867515150552
    # Best params is {'lr': 0.00590263467583827, 'beta1': 0.9947645411289673, 'beta2': 0.9913162240085669} with value 0.007222609438199095


if __name__ == "__main__":
    main()

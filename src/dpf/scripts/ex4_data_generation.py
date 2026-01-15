"""
copied from from https://github.com/Grid2op/lightsim2grid/blob/master/benchmarks/benchmark_grid_size.py
"""

import datetime
import re
import shutil
import time
import warnings
import copy

import lightsim2grid
import pandapower as pp
import numpy as np
from pathlib import Path
from scipy.interpolate import interp1d
from grid2op import make, Parameters
from grid2op.Chronics import FromNPY
from grid2op.Backend import PandaPowerBackend
from lightsim2grid import LightSimBackend
from scipy import sparse

try:
    from lightsim2grid import ContingencyAnalysis
except ImportError:
    from lightsim2grid import SecurityAnalysis as ContingencyAnalysis

from tqdm import tqdm
import os

try:
    from tabulate import tabulate

    TABULATE_AVAIL = True
except ImportError:
    print("The tabulate package is not installed. Some output might not work properly")
    TABULATE_AVAIL = False

VERBOSE = True
MAKE_PLOT = False
WITH_PP = False
DEBUG = False


def get_env_name_displayed(env_name):
    res = re.sub("^l2rpn_", "", env_name)
    res = re.sub("_small$", "", res)
    res = re.sub("_large$", "", res)
    res = re.sub("\\.json$", "", res)
    return res


case_names = [
    "case14.json",
    "case118.json",
    "case_illinois200.json",
    "case300.json",
    "case1354pegase.json",
    "case1888rte.json",
    #   "GBnetwork.json",  # 2224 buses
    "case2848rte.json",
    "case2869pegase.json",
    "case3120sp.json",
    "case6495rte.json",
    "case6515rte.json",
    "case9241pegase.json"
]

solver_names = {lightsim2grid.SolverType.GaussSeidel: "GS",
                lightsim2grid.SolverType.GaussSeidelSynch: "GS synch",
                lightsim2grid.SolverType.SparseLU: "NR (SLU)",
                lightsim2grid.SolverType.KLU: "NR (KLU)",
                lightsim2grid.SolverType.NICSLU: "NR (NICSLU *)",
                lightsim2grid.SolverType.CKTSO: "NR (CKTSO *)",
                lightsim2grid.SolverType.SparseLUSingleSlack: "NR single (SLU)",
                lightsim2grid.SolverType.KLUSingleSlack: "NR single (KLU)",
                lightsim2grid.SolverType.NICSLUSingleSlack: "NR single (NICSLU *)",
                lightsim2grid.SolverType.CKTSOSingleSlack: "NR single (CKTSO *)",
                lightsim2grid.SolverType.FDPF_XB_SparseLU: "FDPF XB (SLU)",
                lightsim2grid.SolverType.FDPF_BX_SparseLU: "FDPF BX (SLU)",
                lightsim2grid.SolverType.FDPF_XB_KLU: "FDPF XB (KLU)",
                lightsim2grid.SolverType.FDPF_BX_KLU: "FDPF BX (KLU)",
                lightsim2grid.SolverType.FDPF_XB_NICSLU: "FDPF XB (NICSLU *)",
                lightsim2grid.SolverType.FDPF_BX_NICSLU: "FDPF BX (NICSLU *)",
                lightsim2grid.SolverType.FDPF_XB_CKTSO: "FDPF XB (CKTSO *)",
                lightsim2grid.SolverType.FDPF_BX_CKTSO: "FDPF BX (CKTSO *)",
                # lightsim2grid.SolverType.DC: "LS+DC",
                # lightsim2grid.SolverType.KLUDC: "LS+SLU",
                # lightsim2grid.SolverType.NICSLUDC: "LS+SLU"
                }


def print_configuration():
    res = []
    print()
    tmp = f"- date: {datetime.datetime.now():%Y-%m-%d %H:%M %z} {time.localtime().tm_zone}"
    res.append(tmp)
    print(tmp)
    try:
        import platform
        tmp = f"- system: {platform.system()} {platform.release()}"
        res.append(tmp)
        print(tmp)
    except ImportError:
        tmp = f"- system: please install the `platform` to have this information"
        res.append(tmp)
        print(tmp)

    try:
        import distro
        tmp = (f"- OS: {distro.linux_distribution(full_distribution_name=False)[0]} "
               f"{distro.linux_distribution(full_distribution_name=False)[1]}")
        res.append(tmp)
        print(tmp)
    except ImportError:
        tmp = (f"- OS: please install the `distro` to have this information")
        res.append(tmp)
        print(tmp)

    try:
        import cpuinfo
        info_ = cpuinfo.get_cpu_info()
        tmp = (f"- processor: {info_['brand_raw']}")
        res.append(tmp)
        print(tmp)
        tmp = (f"- python version: {info_['python_version']}")
        res.append(tmp)
        print(tmp)

    except ImportError:
        tmp = (f"- processor: please install the `py-cpuinfo` to have this information")
        res.append(tmp)
        print(tmp)
        tmp = (f"- python version: please install the `py-cpuinfo` to have this information")
        res.append(tmp)
        print(tmp)

    import pandas as pd
    import pandapower as pp
    import lightsim2grid
    import grid2op
    tmp = (f"- numpy version: {np.__version__}")
    res.append(tmp)
    print(tmp)
    tmp = (f"- pandas version: {pd.__version__}")
    res.append(tmp)
    print(tmp)
    tmp = (f"- pandapower version: {pp.__version__}")
    res.append(tmp)
    print(tmp)
    tmp = (f"- grid2op version: {grid2op.__version__}")
    res.append(tmp)
    print(tmp)
    tmp = (f"- lightsim2grid version: {lightsim2grid.__version__}")
    res.append(tmp)
    print(tmp)
    try:
        from lightsim2grid import compilation_options
        tmp = (f"- lightsim2grid extra information: ")
        res.append(tmp)
        print(tmp)
        print()
        tmp = (f"\t- klu_solver_available: {lightsim2grid.compilation_options.klu_solver_available} ")
        res.append(tmp)
        print(tmp)
        tmp = (f"\t- nicslu_solver_available: {lightsim2grid.compilation_options.nicslu_solver_available} ")
        res.append(tmp)
        print(tmp)
        tmp = (f"\t- cktso_solver_available: {lightsim2grid.compilation_options.cktso_solver_available} ")
        res.append(tmp)
        print(tmp)
        tmp = (f"\t- compiled_march_native: {lightsim2grid.compilation_options.compiled_march_native} ")
        res.append(tmp)
        print(tmp)
        tmp = (f"\t- compiled_o3_optim: {lightsim2grid.compilation_options.compiled_o3_optim} ")
        res.append(tmp)
        print(tmp)
    except ImportError:
        # before it was introduced
        pass
    print()
    return '\n'.join(res)


def make_grid2op_env(pp_case, case_name, load_p, load_q, gen_p, sgen_p):
    param = Parameters.Parameters()
    param.init_from_dict({"NO_OVERFLOW_DISCONNECTION": True})

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        env_lightsim = make("blank",
                            param=param, test=True,
                            backend=LightSimBackend(),
                            chronics_class=FromNPY,
                            data_feeding_kwargs={"load_p": load_p,
                                                 "load_q": load_q,
                                                 "prod_p": gen_p
                                                 },
                            grid_path=f"data/ex4_cases/{case_name}",
                            _add_to_name=f"{case_name}",
                            )
    return env_lightsim


def make_grid2op_env_pp(pp_case, case_name, load_p, load_q, gen_p, sgen_p):
    param = Parameters.Parameters()
    param.init_from_dict({"NO_OVERFLOW_DISCONNECTION": True})

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        env_pp = make("blank",
                      param=param, test=True,
                      backend=PandaPowerBackend(lightsim2grid=False),
                      chronics_class=FromNPY,
                      data_feeding_kwargs={"load_p": load_p,
                                           "load_q": load_q,
                                           "prod_p": gen_p
                                           },
                      grid_path=f"data/ex4_cases/{case_name}",
                      _add_to_name=f"{case_name}",
                      )
    return env_pp


def get_loads_gens(load_p_init, load_q_init, gen_p_init, sgen_p_init, prng):
    # scale loads

    # use some French time series data for loads
    # see https://github.com/BDonnot/data_generation for where to find this file
    coeffs = {"sources": {
        "country": "France",
        "year": "2012",
        "web": "http://clients.rte-france.com/lang/fr/visiteurs/vie/vie_stats_conso_inst.jsp"
    },
        "month": {
            "jan": 1.21,
            "feb": 1.40,
            "mar": 1.05,
            "apr": 1.01,
            "may": 0.86,
            "jun": 0.84,
            "jul": 0.84,
            "aug": 0.79,
            "sep": 0.85,
            "oct": 0.94,
            "nov": 1.01,
            "dec": 1.20
        },
        "day": {
            "mon": 1.01,
            "tue": 1.05,
            "wed": 1.05,
            "thu": 1.05,
            "fri": 1.03,
            "sat": 0.93,
            "sun": 0.88
        },
        "hour": {
            "00:00": 1.00,
            "01:00": 0.93,
            "02:00": 0.91,
            "03:00": 0.86,
            "04:00": 0.84,
            "05:00": 0.85,
            "06:00": 0.90,
            "07:00": 0.97,
            "08:00": 1.03,
            "09:00": 1.06,
            "10:00": 1.08,
            "11:00": 1.09,
            "12:00": 1.09,
            "13:00": 1.09,
            "14:00": 1.06,
            "15:00": 1.03,
            "16:00": 1.00,
            "17:00": 1.00,
            "18:00": 1.04,
            "19:00": 1.09,
            "20:00": 1.05,
            "21:00": 1.01,
            "22:00": 0.99,
            "23:00": 1.03
        }
    }
    vals = list(coeffs["hour"].values())
    x_final = np.arange(12 * len(vals))

    # interpolate them at 5 minutes resolution (instead of 1h)
    vals.append(vals[0])
    vals = np.array(vals) * coeffs["month"]["oct"] * coeffs["day"]["mon"]
    x_interp = 12 * np.arange(len(vals))
    coeffs = interp1d(x=x_interp, y=vals, kind="cubic")
    all_vals = coeffs(x_final).reshape(-1, 1)
    if DEBUG:
        all_vals[:] = 1

    # compute the "smooth" loads matrix
    load_p_smooth = all_vals * load_p_init.reshape(1, -1)
    load_q_smooth = all_vals * load_q_init.reshape(1, -1)

    # add a bit of noise to it to get the "final" loads matrix
    load_p = load_p_smooth * prng.lognormal(mean=0., sigma=0.003, size=load_p_smooth.shape)
    load_q = load_q_smooth * prng.lognormal(mean=0., sigma=0.003, size=load_q_smooth.shape)
    if DEBUG:
        load_p[:] = load_p_smooth
        load_q[:] = load_q_smooth

    # scale generators accordingly
    gen_p = load_p.sum(axis=1).reshape(-1, 1) / load_p_init.sum() * gen_p_init.reshape(1, -1)
    sgen_p = load_p.sum(axis=1).reshape(-1, 1) / load_p_init.sum() * sgen_p_init.reshape(1, -1)
    return load_p, load_q, gen_p, sgen_p


def run_grid2op_env(env_lightsim, case, reset_solver,
                    solver_preproc_solver_time,
                    g2op_speeds,
                    g2op_step_time,
                    ls_solver_time,
                    ls_gridmodel_time,
                    g2op_sizes,
                    sgen_p,
                    nb_ts,
                    store_observations=True,
                    nb_steps=10,
                    store_as_sparse=True
                    ):
    _ = env_lightsim.reset()
    env_name = get_env_name_displayed(env_lightsim.name)
    done = False
    nb_step = 0
    changed_sgen = case.sgen["in_service"].values

    # Init dataset with zeros
    VARIABLES = ("prod_p", "prod_v", "load_p", "load_q", "line_status", "topo_vect",
                 "a_or", "a_ex", "p_or", "p_ex", "q_or", "q_ex", "prod_q", "load_v",
                 "v_or", "v_ex", "theta_or", "theta_ex")

    ALL_VARIABLES = ("prod_p", "prod_v", "load_p", "load_q", "line_status", "topo_vect",
                     "a_or", "a_ex", "p_or", "p_ex", "q_or", "q_ex", "prod_q", "load_v",
                     "v_or", "v_ex", "theta_or", "theta_ex", "SBus", "PV_nodes", "slack", "YBus")

    data = {}
    n_bus_bars = env_lightsim.current_obs.n_sub * 2

    if store_observations:
        for attr_nm in VARIABLES:
            array = getattr(env_lightsim.current_obs, attr_nm)
            data[attr_nm] = np.zeros((nb_steps, array.shape[0]), dtype=array.dtype)

        data["SBus"] = np.zeros((nb_steps, n_bus_bars), dtype=np.complex128)
        data["PV_nodes"] = np.zeros((nb_steps, n_bus_bars), dtype=bool)
        data["slack"] = np.zeros((nb_steps, 2), dtype=np.float16)

        # store the Ybus as sparse matrix, mandatory for large envs as the memory is highly impacted by this matrix
        if store_as_sparse:
            ybus_data = []
            row_indices = []
            col_indices = []
        else:
            data["YBus"] = np.zeros((nb_steps, n_bus_bars, n_bus_bars),
                                    dtype=np.complex128)  # for admittance matrix

    while not done:
        # hack for static gen...
        changed_sgen = copy.deepcopy(case.sgen["in_service"].values)
        this_sgen = sgen_p[nb_step, :].astype(np.float32)
        # this_sgen = sgen_p_init[changed_sgen].astype(np.float32)
        env_lightsim.backend._grid.update_sgens_p(changed_sgen, this_sgen)
        obs, reward, done, info = env_lightsim.step(env_lightsim.action_space())

        if store_observations:
            grid = env_lightsim.backend._grid

            # normal attributes
            for attr_nm in VARIABLES:
                array = getattr(obs, attr_nm)
                data[attr_nm][nb_step, :] = array

            # store physics attributes, see powergridDataSet.py-->_store_physics()
            nb_bus, unique_bus, bus_or, bus_ex = obs._aux_fun_get_bus()
            n_bus_bars = obs._obs_env.n_sub * 2
            admittance_matrix = np.zeros(shape=(n_bus_bars, n_bus_bars), dtype=np.complex128)
            Injection_vect = np.zeros(shape=(n_bus_bars), dtype=np.complex128)
            pv_nodes = np.zeros(shape=(n_bus_bars), dtype=bool)
            YBus = grid.get_Ybus_solver()
            Sbus = grid.get_Sbus_solver()
            admittance_matrix[np.ix_(unique_bus, unique_bus)] = YBus.todense()
            Injection_vect[unique_bus] = Sbus
            pv_nodes[unique_bus[grid.get_pv_solver()]] = True
            prod_bus, _ = obs._get_bus_id(obs.gen_pos_topo_vect, obs.gen_to_subid)
            node_slack_id = prod_bus[-1]
            index_gens_slack = (prod_bus == node_slack_id)
            adjusted_prod_slack = obs.gen_p[index_gens_slack].sum() - Sbus[node_slack_id].real

            if store_as_sparse:
                array_2d = admittance_matrix.reshape(1, -1)
                row_index, col_index = np.nonzero(array_2d)
                data_ = array_2d[row_index, col_index]
                row_indices.extend(row_index + nb_step)
                col_indices.extend(col_index)
                ybus_data.extend(data_)
            else:
                data["YBus"][nb_step, :] = admittance_matrix
            data["SBus"][nb_step, :] = Injection_vect
            data["PV_nodes"][nb_step, :] = pv_nodes
            data["slack"][nb_step, :] = np.array([node_slack_id, adjusted_prod_slack], dtype=np.float16)

            if store_as_sparse:
                sparse_matrix = sparse.csr_matrix((ybus_data, (row_indices, col_indices)),
                                                  shape=(nb_steps, n_bus_bars * n_bus_bars))
                data["YBus"] = sparse_matrix

        if reset_solver:
            env_lightsim.backend._grid.tell_solver_need_reset()
        nb_step += 1
        if nb_step == nb_steps:
            done = True

    # print(data)
    # print(data.keys())

    #print(os.path.join("differentiable_powerflow", "out", "ex4", env_name + ".nz"))
    os.makedirs("data/ex4_data", exist_ok=True)
    SAVE_PATH = os.path.join("data/ex4_data", env_name)

    # create folder
    if not os.path.exists(os.path.abspath(SAVE_PATH)):
        os.mkdir(os.path.abspath(SAVE_PATH))

    if os.path.exists(SAVE_PATH):
        shutil.rmtree(SAVE_PATH)
    os.mkdir(SAVE_PATH)

    for attr_nm in VARIABLES:
        np.savez_compressed(f"{os.path.join(SAVE_PATH, attr_nm)}.npz", data=data[attr_nm])

    if ("YBus" in data.keys()):
        if store_as_sparse:
            sparse.save_npz(os.path.join(SAVE_PATH, "YBus") + ".npz", matrix=data["YBus"])
        else:
            np.savez_compressed(os.path.join(SAVE_PATH, "YBus") + ".npz", data=data["YBus"])
    if ("SBus" in data.keys()):
        np.savez_compressed(os.path.join(SAVE_PATH, "SBus") + ".npz", data=data["SBus"])
    if ("PV_nodes" in data.keys()):
        np.savez_compressed(os.path.join(SAVE_PATH, "PV_nodes") + ".npz", data=data["PV_nodes"])
    if ("slack" in data.keys()):
        np.savez_compressed(os.path.join(SAVE_PATH, "slack") + ".npz", data=data["slack"])

    # NB lightsim2grid does not handle "static gen" because I cannot set "p" in gen in grid2op
    # so results will vary between TimeSeries and grid2op !
    # env_lightsim.backend._grid.tell_solver_need_reset()
    # env_lightsim.backend._grid.dc_pf(env_lightsim.backend.V, 1, 1e-7)
    # env_lightsim.backend._grid.get_bus_status()
    if nb_step != nb_ts:
        #warnings.warn(
        #    f"only able to make {nb_step} (out of {nb_ts}) for {case_name} in grid2op. Results will not be availabe for grid2op step")
        solver_preproc_solver_time.append(None)
        g2op_speeds.append(None)
        g2op_step_time.append(None)
        ls_solver_time.append(None)
        ls_gridmodel_time.append(None)
    else:
        total_time = env_lightsim.backend._timer_preproc + env_lightsim.backend._timer_solver  # + env_lightsim.backend._timer_postproc
        # total_time = env_lightsim._time_step
        solver_preproc_solver_time.append(total_time)
        g2op_speeds.append(1.0 * nb_step / total_time)
        g2op_step_time.append(1.0 * env_lightsim._time_step / nb_step)
        ls_solver_time.append(env_lightsim.backend.comp_time)
        ls_gridmodel_time.append(env_lightsim.backend.timer_gridmodel_xx_pf)
    g2op_sizes.append(env_lightsim.n_sub)
    return nb_step


def main():
    prng = np.random.default_rng(42)
    case_names_displayed = [get_env_name_displayed(el) for el in case_names]
    solver_preproc_solver_time = []
    g2op_speeds = []
    g2op_sizes = []
    g2op_step_time = []
    ls_solver_time = []
    ls_gridmodel_time = []

    solver_preproc_solver_time_reset = []
    g2op_speeds_reset = []
    g2op_sizes_reset = []
    g2op_step_time_reset = []
    ls_solver_time_reset = []
    ls_gridmodel_time_reset = []

    ts_times = []
    ts_speeds = []
    ts_sizes = []
    sa_times = []
    sa_speeds = []
    sa_sizes = []

    os.makedirs(f"data/ex4_cases", exist_ok=True)

    for case_name in tqdm(case_names):
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

        res_time = 1.
        res_unit = "s"
        if len(load_p_init) <= 1000:
            res_time = 1e3
            res_unit = "ms"

        # simulate the data
        load_p, load_q, gen_p, sgen_p = get_loads_gens(load_p_init, load_q_init, gen_p_init, sgen_p_init, prng)
        nb_ts = gen_p.shape[0]
        # add slack !
        slack_gens = np.zeros((nb_ts, case.ext_grid.shape[0]))
        if "res_ext_grid" in case:
            slack_gens += np.tile(case.res_ext_grid["p_mw"].values.reshape(1, -1), (nb_ts, 1))
        gen_p_g2op = np.concatenate((gen_p, slack_gens), axis=1)

        env_lightsim = make_grid2op_env(case,
                                        case_name,
                                        load_p,
                                        load_q,
                                        gen_p_g2op,
                                        sgen_p)
        # Perform the computation using grid2op

        print(env_lightsim.env_name)

        reset_solver = False  # default
        nb_step = run_grid2op_env(env_lightsim, case, reset_solver,
                                  solver_preproc_solver_time,
                                  g2op_speeds,
                                  g2op_step_time,
                                  ls_solver_time,
                                  ls_gridmodel_time,
                                  g2op_sizes, sgen_p, nb_ts,
                                  store_observations=True,
                                  nb_steps=10
                                  )

        ts_sizes.append(env_lightsim.n_sub)
        env_lightsim.close()


if __name__ == "__main__":
    main()

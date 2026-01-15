"""
This file generates a Time Series dataset where the grid is unchanged but the injections vary over time.
We use the IEEE-118 grid for this purpose (similar to LIPS).
Useful sources:
https://github.com/Grid2op/lightsim2grid/blob/master/benchmarks/benchmark_grid_size.py
LIPS repository
"""
import grid2op

"""
copied and adapted from from https://github.com/Grid2op/lightsim2grid/blob/master/benchmarks/benchmark_grid_size.py
"""

import shutil

import numpy as np
from lightsim2grid import LightSimBackend
from scipy import sparse
from grid2op.Agent.doNothing import DoNothingAgent

try:
    from lightsim2grid import ContingencyAnalysis
except ImportError:
    from lightsim2grid import SecurityAnalysis as ContingencyAnalysis

import os

def main():
    # two options,
    # 1) Time-Series module: given V_0 and injection-series let computer compute flows
    # 2) create env and run grid2op with non-actions and store states
    # 2 is better for us
    # see https://grid2op.readthedocs.io/en/latest/user/chronics.html for time-series in grid2op

    env_name = "l2rpn_idf_2023"
    env = grid2op.make(env_name, backend=LightSimBackend())
    obs = env.reset(options={"time serie id": 0}, seed=0)
    # the time series id determines which injection/time_series we start with
    # the seed determines the randomness of the opponent, maintenance ...
    my_agent = DoNothingAgent(env.action_space)
    reward = env.reward_range[0]
    done = False
    nb_steps = 500  # after running it once (there are 514 steps in this episode)
    store_as_sparse = True
    reset_solver = False

    #  grid = env.backend._grid
    # YBus = grid.get_Ybus_solver()
    # Sbus = grid.get_Sbus_solver()
    # print(Sbus[:2])

    VARIABLES = ("prod_p", "prod_v", "load_p", "load_q", "line_status", "topo_vect",
                 "a_or", "a_ex", "p_or", "p_ex", "q_or", "q_ex", "prod_q", "load_v",
                 "v_or", "v_ex", "theta_or", "theta_ex")

    ALL_VARIABLES = ("prod_p", "prod_v", "load_p", "load_q", "line_status", "topo_vect",
                     "a_or", "a_ex", "p_or", "p_ex", "q_or", "q_ex", "prod_q", "load_v",
                     "v_or", "v_ex", "theta_or", "theta_ex", "SBus", "PV_nodes", "slack", "YBus")

    FIXED_VARIABLES = ("line_status", "topo_vect", "PV_nodes", "YBus")

    data = {}
    n_bus_bars = env.current_obs.n_sub * 2

    for attr_nm in VARIABLES:
        array = getattr(env.current_obs, attr_nm)
        if attr_nm in FIXED_VARIABLES:
            data[attr_nm] = np.zeros((1, array.shape[0]), dtype=array.dtype)  # only save once
        else:
            data[attr_nm] = np.zeros((nb_steps, array.shape[0]), dtype=array.dtype)

    data["SBus"] = np.zeros((nb_steps, n_bus_bars), dtype=np.complex128)
    data["PV_nodes"] = np.zeros((1, n_bus_bars), dtype=bool)
    data["slack"] = np.zeros((nb_steps, 2), dtype=np.float16)

    # store the Ybus as sparse matrix, mandatory for large envs as the memory is highly impacted by this matrix
    if store_as_sparse:
        ybus_data = []
        row_indices = []
        col_indices = []
    else:
        data["YBus"] = np.zeros((1, n_bus_bars, n_bus_bars),
                                dtype=np.complex128)

    nb_step = 0
    while not done:
        act = my_agent.act(obs, reward, done)
        obs, reward, done, info = env.step(act)  # we can ignore obs here since we access the env directly

        grid = env.backend._grid

        # normal attributes
        for attr_nm in VARIABLES:
            array = getattr(obs, attr_nm)
            if attr_nm in FIXED_VARIABLES:
                data[attr_nm][0, :] = array
            else:
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
            if nb_step == 0:
                array_2d = admittance_matrix.reshape(1, -1)
                row_index, col_index = np.nonzero(array_2d)
                data_ = array_2d[row_index, col_index]
                row_indices.extend(row_index + nb_step)
                col_indices.extend(col_index)
                ybus_data.extend(data_)
        else:
            if nb_step == 0:
                data["YBus"][nb_step, :] = admittance_matrix
        data["SBus"][nb_step, :] = Injection_vect
        if nb_step == 0:
            data["PV_nodes"][nb_step, :] = pv_nodes
        data["slack"][nb_step, :] = np.array([node_slack_id, adjusted_prod_slack], dtype=np.float16)

        if store_as_sparse:
            if nb_step == 0:
                sparse_matrix = sparse.csr_matrix((ybus_data, (row_indices, col_indices)),
                                                  shape=(1, n_bus_bars * n_bus_bars))
                data["YBus"] = sparse_matrix

        if reset_solver:
            env.backend._grid.tell_solver_need_reset()
        nb_step += 1
        if nb_step == nb_steps:
            done = True

    # save the data
    os.makedirs(f"data/ex5_data", exist_ok=True)
    SAVE_PATH = os.path.join("data", "ex5_data", env_name)
    # differentiable_powerflow / out / ex5 / <env_name>.nz

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
    if "SBus" in data.keys():
        np.savez_compressed(os.path.join(SAVE_PATH, "SBus") + ".npz", data=data["SBus"])
    if ("PV_nodes" in data.keys()):
        np.savez_compressed(os.path.join(SAVE_PATH, "PV_nodes") + ".npz", data=data["PV_nodes"])
    if ("slack" in data.keys()):
        np.savez_compressed(os.path.join(SAVE_PATH, "slack") + ".npz", data=data["slack"])

if __name__ == "__main__":
    main()

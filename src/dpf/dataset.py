import os
import os
import numpy as np
from scipy import sparse

from lips.benchmark.powergridBenchmark import PowerGridBenchmark
from lips.dataset.powergridDataSet import downloadPowergridDataset

parent_dir = os.pardir

benchmark_kwargs = {"attr_x": ("prod_p", "prod_v", "load_p", "load_q"),
                    "attr_y": ("a_or", "a_ex", "p_or", "p_ex", "v_or", "v_ex", "theta_or", "theta_ex"),
                    "attr_tau": ("line_status", "topo_vect"),
                    "attr_physics": ('YBus', 'SBus', 'PV_nodes', 'slack')}


class LipsDataset:

    def __init__(self, load_data=False, download_data=False, generate_data=False,
                 load_ybus_as_as_sparse=True, env_name="lips_idf_2023"):
        self.benchmark = None
        self.BENCH_CONFIG_PATH = os.path.join("data/configs", "benchmarks", env_name + ".ini")
        self.DATA_PATH = os.path.join(f"data/" + "input_data_local", env_name)
        self.LOG_PATH = f"data/logs/" + env_name + "_log.log"

        os.makedirs(f"data/input_data_local/{env_name}/Benchmark_competition", exist_ok=True)
        os.makedirs(f"data/logs", exist_ok=True)

        if download_data:
            downloadPowergridDataset(os.path.join(f"data/", "input_data_local"), env_name)

        if load_data or download_data:
            self.benchmark = PowerGridBenchmark(benchmark_path=self.DATA_PATH,
                                                config_path=self.BENCH_CONFIG_PATH,
                                                benchmark_name="Benchmark_competition",
                                                load_data_set=load_data,
                                                load_ybus_as_sparse=load_ybus_as_as_sparse,
                                                log_path=self.LOG_PATH,
                                                )

        if generate_data:
            self.benchmark = PowerGridBenchmark(benchmark_path=self.DATA_PATH,
                                                config_path=self.BENCH_CONFIG_PATH,
                                                benchmark_name="Benchmark_competition",
                                                load_data_set=False,  # to load already generated dataset
                                                load_ybus_as_sparse=load_ybus_as_as_sparse,
                                                # Ybus is registered as sparse
                                                log_path=self.LOG_PATH,
                                                )

            self.benchmark.generate(nb_sample_train=int(3e5), nb_sample_val=int(1e5), nb_sample_test=int(1e5),
                                    nb_sample_test_ood_topo=int(2e5), do_store_physics=True,
                                    store_as_sparse=load_ybus_as_as_sparse)


        self.train_dataset = self.benchmark.train_dataset
        self.valid_dataset = self.benchmark.val_dataset
        self.test_dataset = self.benchmark._test_dataset
        self.test_ood_dataset = self.benchmark._test_ood_topo_dataset

    def get_sample(self, dataset, sample_id):
        # Input variables
        # data_this = benchmark.train_dataset.get_data(np.array([sample], dtype=int))

        data = dataset.data

        prod_p = data["prod_p"][sample_id]  # production active power, input variable
        prod_v = data["prod_v"][sample_id]  # production voltage, input variable
        load_p = data["load_p"][sample_id]  # load active power, input variable
        load_q = data["load_q"][sample_id]  # load reactive power, input variable

        line_status = data["line_status"][sample_id]  # connected or disconnected, redundant information!
        topo_vect = data["topo_vect"][
            sample_id]  # for each element (power line, production,load,storage): 1,2 (connected) or -1 (disconnected)

        Ybus = data["YBus"][sample_id]  # admittance matrix
        Sbus = data["SBus"][sample_id]  # complex injections at each bus (active power and reactive power)
        PV_nodes = data["PV_nodes"][sample_id]  # specified (active) P(ower) and V(oltage magnitude) (generators?)
        slack = data["slack"][sample_id]  # reference bus: voltage magnitude in kV and voltage angle in degrees?

        # target variables
        a_or = data["a_or"][sample_id]  # currents at origin of power-lines, target variable
        a_ex = data["a_ex"][sample_id]  # currents at extremity, target variable
        p_or = data["p_or"][sample_id]  # active power at origin, target variable
        p_ex = data["p_ex"][sample_id]  # active power at extremity, target variable
        v_or = data["v_or"][sample_id]  # voltages at origin, target variable
        v_ex = data["v_ex"][sample_id]  # voltages at extremity, target variable
        theta_or = data["theta_or"][
            sample_id]  # voltage angle at origin, optional target variable (other target variables can infer these)
        theta_ex = data["theta_ex"][sample_id]  # voltage angle at extremity, optional target variable (..)

        inputs = prod_p, prod_v, load_p, load_q, line_status, topo_vect, Ybus, Sbus, PV_nodes, slack
        targets = a_or, a_ex, p_or, p_ex, v_or, v_ex, theta_or, theta_ex
        return inputs, targets


ALL_VARIABLES = ("prod_p", "prod_v", "load_p", "load_q", "line_status", "topo_vect",
                 "a_or", "a_ex", "p_or", "p_ex", "q_or", "q_ex", "prod_q", "load_v",
                 "v_or", "v_ex", "theta_or", "theta_ex", "SBus", "PV_nodes", "slack", "YBus")


class CustomGridDataset:
    def __init__(self, env_name="case2848rte", load_ybus_as_sparse=True):
        self.SAVE_PATH = os.path.join("data", "ex4_data", "blank" + env_name)
        self.data = {}
        # load data

        for attr_nm in ALL_VARIABLES:
            path_this_array = f"{os.path.join(self.SAVE_PATH, attr_nm)}.npz"
            if (attr_nm == "YBus") and (load_ybus_as_sparse):
                self.data[attr_nm] = sparse.load_npz(path_this_array)
            else:
                self.data[attr_nm] = np.load(path_this_array)["data"]

    def get_sample(self, sample_id):
        # Input variables
        # data_this = benchmark.train_dataset.get_data(np.array([sample], dtype=int))

        data = self.data
        prod_p = data["prod_p"][sample_id]  # production active power, input variable
        prod_v = data["prod_v"][sample_id]  # production voltage, input variable
        load_p = data["load_p"][sample_id]  # load active power, input variable
        load_q = data["load_q"][sample_id]  # load reactive power, input variable

        line_status = data["line_status"][sample_id]  # connected or disconnected, redundant information!
        topo_vect = data["topo_vect"][
            sample_id]  # for each element (power line, production,load,storage): 1,2 (connected) or -1 (disconnected)

        Ybus = data["YBus"][sample_id]  # admittance matrix
        Sbus = data["SBus"][sample_id]  # complex injections at each bus (active power and reactive power)
        PV_nodes = data["PV_nodes"][sample_id]  # specified (active) P(ower) and V(oltage magnitude) (generators?)
        slack = data["slack"][sample_id]  # reference bus: voltage magnitude in kV and voltage angle in degrees?

        # target variables
        a_or = data["a_or"][sample_id]  # currents at origin of power-lines, target variable
        a_ex = data["a_ex"][sample_id]  # currents at extremity, target variable
        p_or = data["p_or"][sample_id]  # active power at origin, target variable
        p_ex = data["p_ex"][sample_id]  # active power at extremity, target variable
        v_or = data["v_or"][sample_id]  # voltages at origin, target variable
        v_ex = data["v_ex"][sample_id]  # voltages at extremity, target variable
        theta_or = data["theta_or"][
            sample_id]  # voltage angle at origin, optional target variable (other target variables can infer these)
        theta_ex = data["theta_ex"][sample_id]  # voltage angle at extremity, optional target variable (..)

        inputs = prod_p, prod_v, load_p, load_q, line_status, topo_vect, Ybus, Sbus, PV_nodes, slack
        targets = a_or, a_ex, p_or, p_ex, v_or, v_ex, theta_or, theta_ex
        return inputs, targets


class SmallTimeSeriesDataset:
    def __init__(self, load_ybus_as_sparse=True):
        self.SAVE_PATH = os.path.join("data", "ex5_data", "l2rpn_idf_2023")
        self.data = {}
        # load data

        for attr_nm in ALL_VARIABLES:
            path_this_array = f"{os.path.join(self.SAVE_PATH, attr_nm)}.npz"
            if (attr_nm == "YBus") and (load_ybus_as_sparse):
                self.data[attr_nm] = sparse.load_npz(path_this_array)
            else:
                self.data[attr_nm] = np.load(path_this_array)["data"]

    def get_fixed_attributes(self):
        data = self.data
        line_status = data["line_status"][0]
        topo_vect = data["topo_vect"][0]
        Ybus = data["YBus"][0]
        PV_nodes = data["PV_nodes"][0]
        slack_id = data["slack"][0][0]  # [slack id, adjusted_prod_slack]

        return line_status, topo_vect, Ybus, PV_nodes, slack_id

    def get_injections(self):
        data = self.data

        prod_p = data["prod_p"][:]  # production active power, input variable
        prod_v = data["prod_v"][:]  # production voltage, input variable
        load_p = data["load_p"][:]  # load active power, input variable
        load_q = data["load_q"][:]  # load reactive power, input variable

        Sbus = data["SBus"][:]  # complex injections at each bus (active power and reactive power)

        # target variables
        a_or = data["a_or"][:]  # currents at origin of power-lines, target variable
        a_ex = data["a_ex"][:]  # currents at extremity, target variable
        p_or = data["p_or"][:]  # active power at origin, target variable
        p_ex = data["p_ex"][:]  # active power at extremity, target variable
        v_or = data["v_or"][:]  # voltages at origin, target variable
        v_ex = data["v_ex"][:]  # voltages at extremity, target variable
        theta_or = data["theta_or"][
                   :]  # voltage angle at origin, optional target variable (other target variables can infer these)
        theta_ex = data["theta_ex"][:]  # voltage angle at extremity, optional target variable (..)

        inputs = prod_p, prod_v, load_p, load_q, Sbus
        targets = a_or, a_ex, p_or, p_ex, v_or, v_ex, theta_or, theta_ex
        return inputs, targets

    def get_injection(self, sample_id):
        data = self.data

        prod_p = data["prod_p"][sample_id]  # production active power, input variable
        prod_v = data["prod_v"][sample_id]  # production voltage, input variable
        load_p = data["load_p"][sample_id]  # load active power, input variable
        load_q = data["load_q"][sample_id]  # load reactive power, input variable

        Sbus = data["SBus"][sample_id]  # complex injections at each bus (active power and reactive power)

        # target variables
        a_or = data["a_or"][sample_id]  # currents at origin of power-lines, target variable
        a_ex = data["a_ex"][sample_id]  # currents at extremity, target variable
        p_or = data["p_or"][sample_id]  # active power at origin, target variable
        p_ex = data["p_ex"][sample_id]  # active power at extremity, target variable
        v_or = data["v_or"][sample_id]  # voltages at origin, target variable
        v_ex = data["v_ex"][sample_id]  # voltages at extremity, target variable
        theta_or = data["theta_or"][
            sample_id]  # voltage angle at origin, optional target variable (other target variables can infer these)
        theta_ex = data["theta_ex"][sample_id]  # voltage angle at extremity, optional target variable (..)

        inputs = prod_p, prod_v, load_p, load_q, Sbus
        targets = a_or, a_ex, p_or, p_ex, v_or, v_ex, theta_or, theta_ex
        return inputs, targets

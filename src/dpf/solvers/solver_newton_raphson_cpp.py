import time

import numpy as np
import scipy
from lightsim2grid_cpp import KLUSolverSingleSlack
from scipy.sparse import coo_matrix

from dpf.solvers.abstract_powerflow_solver import AbstractPowerFlowSolver


def get_all_zero_entries_without_diagonal(matrix):
    n_rows, n_cols = matrix.shape
    all_indices = set(
        (i, j) for i in range(n_rows) for j in range(n_cols) if i != j
    )  # takes long
    non_zero_set = set(zip(*matrix.nonzero()))
    zero_set = all_indices - non_zero_set  # takes even longer
    return list(
        zero_set
    )  # returns all available edges as a list, converting to a list takes long, stuck here


def get_k_random_zero_entries_without_diagonal(matrix, k, seed=0):
    """

    :param matrix: matrix
    :param k: number of connections added
    :param seed: seed
    :return: symmetric list of new connections (i,j),(j,i)
    """
    # adds k connections /  2k edges

    non_zero_set = set(zip(*matrix.nonzero()))
    n = matrix.shape[0]
    new_indices = []
    added = 0
    np.random.seed(seed)
    while added < k:
        i = np.random.randint(0, n)
        j = np.random.randint(0, n)
        # print(f"trying to add {i,j}")

        if i == j or (i, j) in non_zero_set or (i, j) in new_indices:
            continue
        # assume that (i,j) is a new connection and add it new_indices
        new_indices.append((i, j))
        new_indices.append((j, i))
        added = added + 1

    return new_indices


def get_k_random_values_duplicated(k, mean=0, std=0, seed=0):
    """Currently hardcoded to give k fixed small values near 0 to ensure solvability but add extra run-time from
    the new non-zero entries."""

    def random_complex(mean_, std_):
        sigma = std_ / np.sqrt(2)
        real_part = np.random.normal(loc=mean_.real, scale=sigma)
        imag_part = np.random.normal(loc=mean_.imag, scale=sigma)
        return real_part + 1j * imag_part

    # [z1,z1, z2,z2, ... , zk,zk]
    random_list = []
    for i in range(k):
        # get a random number
        z = 0.00001515 + 0.00001515 * i  # for now add a really smalll fixed number here
        # z = random_complex(mean, std)  # large std leads to larger values here. Solvability suffers from that
        random_list.append(z)
        random_list.append(z)

    return random_list


class NRPowerFlowSolverCPP(AbstractPowerFlowSolver):
    """Power flow solver class calling the LightSim2Grid KLU solver."""

    def __init__(self, backend):
        super().__init__(backend)

    def run_pf(self):
        # lightsim2grid_cpp.KLUSolverSingleSlack
        # do the newton raphson algorithm
        # solver.solve(Ybus, V0, Sbus, ref, slack_weights, pv, pq, max_iteration, tolerance_pu)

        # see in Solvers.h:  typedef BaseNRSingleSlackAlgo<KLULinearSolver> KLUSolverSingleSlack;
        # --> Look at 1) BaseNRSingleSlackAlgo and 2) KLULinearSolver

        # inputs
        Ybus_ = self.Ybus_solver
        V_ = self.V
        Sbus_ = self.Sbus_solver
        slack_ids_solver_ = self.slack_ids_solver
        slack_weights_ = self.slack_weights_solver
        pv_nodes_ = self.pv_nodes_solver
        pq_nodes_ = self.pq_nodes_solver
        max_iteration_ = self.max_iteration_solver
        # max_iteration_ = 1 # debugging
        tolerance_pu_ = self.tolerance_pu_solver

        # use solver
        solver = KLUSolverSingleSlack()
        solver.reset()
        success = solver.solve(
            Ybus_,
            V_,
            Sbus_,
            slack_ids_solver_,
            slack_weights_,
            pv_nodes_,
            pq_nodes_,
            max_iteration_,
            tolerance_pu_,
        )
        print("success: ", success)
        print("iterations needed:", solver.get_nb_iter())

        # print("success: ", success)
        converged = solver.converged()
        iterations = solver.get_nb_iter()
        # print("converged: ", converged)
        # print("iterations: ", iterations)

        # return Voltages and Jacobian
        self.Va = (
            solver.get_Va().copy()
        )  # copy important here! the result is tied to the solver which gets reset
        # print(self.Va)
        self.Vm = (
            solver.get_Vm().copy()
        )  # copy important here! the result is tied to the solver which gets reset
        # print(self.Vm)
        self.V = self.Vm * np.exp(1j * self.Va)  # complex voltage
        # print(self.V)
        self.J = solver.get_J().copy()  # copy important here!
        # print(self.J)
        self.S_calc = self.V * np.conjugate(Ybus_ * self.V)

    def add_new_random_connections_to_ybus(self, nb_new_random_connections, mean, std):
        if nb_new_random_connections == 0:
            return

        # call after init so that Ybus has shape (nb_active_buses, nb_active_buses)

        # https://stackoverflow.com/questions/38241386/what-is-the-correct-way-to-add-elements-to-a-csr-matrix
        # Change index structure only once for efficiency reasons
        # And make lilmatrix?
        #

        # missing_connections = get_all_zero_entries_without_diagonal(self.Ybus_solver)
        # sampled_indices = np.random.choice(len(missing_connections), size=nb_new_random_connections, replace=False)
        # new_indices = [missing_connections[i] for i in sampled_indices]

        # print(self.Ybus_solver)
        # print(self.Ybus_solver[0,2315]) # (-0.11304887257029134+43.623452234030864j)
        # print(self.Ybus_solver[2315, 0]) #  (-0.11304887257029134+43.623452234030864j)

        new_indices = get_k_random_zero_entries_without_diagonal(
            self.Ybus_solver, nb_new_random_connections
        )
        # [(i1,j1), (j1,i1), (i2,j2), (j2,i2), ...]
        new_values = get_k_random_values_duplicated(
            nb_new_random_connections, mean, std
        )
        # TODO new_values is currently hardcoded

        rows, cols = zip(*new_indices)
        shape = self.Ybus_solver.shape
        update = coo_matrix((new_values, (rows, cols)), shape=shape).tocsr()

        Ybus_solver_new = self.Ybus_solver + update
        self.Ybus_solver = Ybus_solver_new

    def run_pf_super_grid(
        self,
        batch_size,
        max_iteration_=1000,
        strategy="no_connections",
        strategy_amount_param=0,
    ):
        Ybus_ = self.Ybus_solver  # sparse csr matrix
        V_ = self.V
        Sbus_ = self.Sbus_solver
        slack_ids_solver_ = self.slack_ids_solver
        slack_weights_ = self.slack_weights_solver
        pv_nodes_ = self.pv_nodes_solver
        pq_nodes_ = self.pq_nodes_solver
        tolerance_pu_ = self.tolerance_pu_solver

        # Ybus batching
        block_list = [Ybus_.copy() for _ in range(batch_size)]
        Ybus_scaled = scipy.sparse.block_diag(block_list, format="csr")

        # V, Sbus
        V_duplicated = np.tile(V_, batch_size)
        Sbus_duplicated = np.tile(Sbus_, batch_size)

        # indices
        pvs_shifted = np.concatenate(
            [pv_nodes_ + i * self.nb_active_buses for i in range(batch_size)]
        )
        pqs_shifted = np.concatenate(
            [pq_nodes_ + i * self.nb_active_buses for i in range(batch_size)]
        )

        # modifications to Ybus / connections
        if strategy == "no_connections":
            # leave Ybus in a block diagonal structure without any connections
            pass
        elif strategy == "total_random":
            # adds batch_size * strategy_amount_param many random connections
            new_indices = get_k_random_zero_entries_without_diagonal(
                Ybus_scaled, strategy_amount_param * batch_size
            )
            # [(i1,j1), (j1,i1), (i2,j2), (j2,i2), ...]
            new_values = get_k_random_values_duplicated(
                batch_size * strategy_amount_param
            )  # hardcoded small values

            rows, cols = zip(*new_indices)
            shape = Ybus_scaled.shape
            update = coo_matrix((new_values, (rows, cols)), shape=shape).tocsr()

            Ybus_scaled_new = Ybus_scaled + update
            Ybus_scaled = Ybus_scaled_new

        elif strategy == "linear_random":
            pass
        elif strategy == "pairwise_random":
            pass
        else:
            pass

        start_time = time.perf_counter()
        # use solver
        solver = KLUSolverSingleSlack()
        solver.reset()
        success = solver.solve(
            Ybus_scaled,
            V_duplicated,
            Sbus_duplicated,
            slack_ids_solver_,
            slack_weights_,
            pvs_shifted,
            pqs_shifted,
            max_iteration_,
            tolerance_pu_,
        )
        end_time = time.perf_counter()

        print("converged: ", success)
        print("iterations used: ", solver.get_nb_iter())
        # print("success: ", success)
        converged = solver.converged()
        iterations = solver.get_nb_iter()
        # print("converged: ", converged)
        # print("iterations: ", iterations)

        run_time = end_time - start_time
        return run_time

    def run_pf_batched(
        self,
        batch_size,
        ybus_scaling_method="block_diagonal",
        density=0.5,
        max_iteration_=1000,
    ):
        # this method is purely to test how fast larger grids are completed
        # methods:
        # ybus_scaling_method = "block_diagonal"
        # ybus_scaling_method = "random_dense" # TODO

        # inputs
        Ybus_ = self.Ybus_solver  # sparse csr matrix
        V_ = self.V
        Sbus_ = self.Sbus_solver
        slack_ids_solver_ = self.slack_ids_solver
        slack_weights_ = self.slack_weights_solver
        pv_nodes_ = self.pv_nodes_solver
        pq_nodes_ = self.pq_nodes_solver
        # max_iteration_ = self.max_iteration_solver
        # max_iteration_ = 6  # this is enough for 9241pegase. More might be needed for random inputs

        # max_iteration_ = 1 # debugging
        tolerance_pu_ = self.tolerance_pu_solver

        # TODO do the batching of the inputs

        # Ybus
        if ybus_scaling_method == "block_diagonal":
            print("Original shape: ", Ybus_.shape)
            print("Original nnz: ", Ybus_.nnz)
            block_list = [Ybus_.copy() for _ in range(batch_size)]
            Ybus_scaled = scipy.sparse.block_diag(block_list, format="csr")
            print(
                "new Ybus shape: ", Ybus_scaled.shape
            )  # (9241 * batchsize, 9241 * batch_size)
            print("nnz of Ybus: ", Ybus_scaled.nnz)  # 37655 * batch_size

        if ybus_scaling_method == "random":
            print("Original shape: ", Ybus_.shape)
            print("Original nnz: ", Ybus_.nnz)
            # new dimensions
            # print(Ybus_.shape)  # (9241,9241)
            # print(Ybus_.dtype)  # complex128
            # print(type(Ybus_[0,0]))  # <class 'numpy.complex128'>
            # print(type(Ybus_)) # scipy sparse csr

            rng = np.random.default_rng(seed=42)
            Ybus_scaled = scipy.sparse.random(
                m=Ybus_.shape[0] * batch_size,
                n=Ybus_.shape[1] * batch_size,
                density=density,
                format="csr",
                dtype=Ybus_.dtype,
                random_state=rng,
            )
            print("new Ybus shape: ", Ybus_scaled.shape)
            print("nnz of Ybus: ", Ybus_scaled.nnz)

        # V, Sbus
        V_duplicated = np.tile(V_, batch_size)
        Sbus_duplicated = np.tile(Sbus_, batch_size)

        # indices
        pvs_shifted = np.concatenate(
            [pv_nodes_ + i * self.nb_active_buses for i in range(batch_size)]
        )
        pqs_shifted = np.concatenate(
            [pq_nodes_ + i * self.nb_active_buses for i in range(batch_size)]
        )

        # slack
        # print(slack_ids_solver_) # 4230, ignored anyways
        # print(slack_weights_) # [0. 0. 0. ... 0. 0. 0.]  , ignored anyways

        start_time = time.perf_counter()
        # use solver
        solver = KLUSolverSingleSlack()
        solver.reset()
        success = solver.solve(
            Ybus_scaled,
            V_duplicated,
            Sbus_duplicated,
            slack_ids_solver_,
            slack_weights_,
            pvs_shifted,
            pqs_shifted,
            max_iteration_,
            tolerance_pu_,
        )
        end_time = time.perf_counter()

        print("converged: ", success)
        print("iterations used: ", solver.get_nb_iter())
        # print("success: ", success)
        converged = solver.converged()
        iterations = solver.get_nb_iter()
        # print("converged: ", converged)
        # print("iterations: ", iterations)
        # return Voltages and Jacobian
        # self.Va = solver.get_Va().copy()  # copy important here! the result is tied to the solver which gets reset
        # print(self.Va)
        # self.Vm = solver.get_Vm().copy()  # copy important here! the result is tied to the solver which gets reset
        # print(self.Vm)
        # self.V = self.Vm * np.exp(1j * self.Va)  # complex voltage
        # print(self.V)
        # self.J = solver.get_J().copy()  # copy important here!
        # print(self.J)
        # self.S_calc = self.V * np.conjugate(Ybus_ * self.V)

        run_time = end_time - start_time
        return run_time

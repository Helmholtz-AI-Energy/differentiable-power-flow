"""
This is just a mock-batched file.
The goal is to find out the speed-up of using batching and a gpu.
So we do not actually use different grids at once but the same one multiple times.
"""

# TODO store relevant states in class and only update injections
"""
This class is made for the purpose of handling time series where the grid stays the same but the injections change.
"""

import time

import numpy as np
from scipy.sparse import coo_matrix
import torch

from dpf.solvers.abstract_powerflow_solver import AbstractPowerFlowSolver


def get_k_random_zero_entries_without_diagonal(matrix, k, seed=0):
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

class TimeSeriesPowerFlowSolverBatched(AbstractPowerFlowSolver):
    def __init__(self, backend, hyperparams=None, continuation_hyperparams=None):
        super().__init__(backend)

        self.device = None
        self.use_gpu = None
        self.Ybus_conj_torch_batched = None
        self.Sbus_torch_batched = None
        self.Vm_fixed_batched = None
        self.Va_fixed_batched = None
        self.Sbus_solver_batched = None
        self.pq_nodes_torch_batched = None
        self.pv_nodes_torch_batched = None
        self.batch_size = None
        self.scheduler = None
        self.optimizer = None
        self.params = None
        self.Vm_learnable = None
        self.Va_learnable = None
        self.optimizer_init_time = None
        self.times_list = None
        self.inf_norm_loss_list = None
        self.Ybus_torch_batched = None
        self.slack_id_torch = None
        self.loss_list = None
        self.best_loss = None
        self.best_checkpoint = None
        if hyperparams is None:
            self.hyperparams = {
                "optimizer_class": torch.optim.Adam,
                "optimizer_kwargs": {"lr": 0.003377, "betas": (0.979681, 0.963442)},
                "scheduler_class": torch.optim.lr_scheduler.ReduceLROnPlateau,
                "scheduler_kwargs": {"factor": 0.547191, "patience": 41, "threshold_mode": "rel",
                                     "threshold": 0.067321, "cooldown": 97},
                "loss_fn": torch.nn.MSELoss(),
                "max_iter": 1000,
                "tol": 1e-8}
        else:
            self.hyperparams = hyperparams
        self.continuation_hyperparams = continuation_hyperparams

    def init_v(self, strategy="ones", random_init_seed=0):
        if strategy == "ones":
            # self.V = np.array(np.ones(self.nb_active_buses, dtype=np.complex128) * self.init_vm_pu_solver)
            self.V = np.array(
                np.ones(self.batch_size * self.nb_active_buses, dtype=np.complex128) * self.init_vm_pu_solver)
        pass

    def preprocess(self, topo_vect, prods_p, prods_v, loads_p, loads_q, Ybus, Sbuses, pv, line_status=None):
        self.fill_backend_with_data(topo_vect, prods_p[0], prods_v[0], loads_p[0], loads_q[0])
        if line_status is not None:
            self.line_status = line_status
        self.fetch_grid_data()
        self.find_active_buses()
        self.init_slack()
        self.fillYbus(Ybus)  # self.Ybus_solver is only for the first batch.
        self.fill_pv_pq(pv)

        assert Sbuses.shape[0] == self.batch_size
        Sbuses_transformed = np.ndarray((self.batch_size * self.nb_active_buses), dtype=np.complex128)

        for i in range(self.batch_size):
            # do indexing for current Sbus and store it in Sbuses_transformed

            # TODO is this line really necessary? probably not..
            self.fill_backend_with_data(topo_vect, prods_p[i], prods_v[i], loads_p[i], loads_q[i])
            current_Sbus = Sbuses[i]
            res = np.ndarray(self.nb_active_buses, dtype=np.complex128)
            for j in range(self.nb_active_buses):
                global_bus_id = self.id_solver_to_me[j]
                res[j] = current_Sbus[global_bus_id]
            current_Sbus_solver = res

            Sbuses_transformed[i * self.nb_active_buses: (i + 1) * self.nb_active_buses] = current_Sbus_solver

        self.Sbus_solver_batched = Sbuses_transformed

    def set_batch_size(self, batch_size):
        self.batch_size = batch_size

    def set_gpu_usage(self, use_gpu=False):
        if use_gpu:
            assert torch.cuda.is_available()
        device = torch.device("cuda" if use_gpu else "cpu")
        self.use_gpu = use_gpu
        self.device = device


    def reconstruct_Va(self, Va_learnable):
        Va_new = self.Va_fixed_batched.clone()
        Va_new[torch.concatenate([self.pv_nodes_torch_batched, self.pq_nodes_torch_batched])] = Va_learnable
        return Va_new

    def reconstruct_Vm(self, Vm_learnable):
        Vm_new = self.Vm_fixed_batched.clone()
        Vm_new[self.pq_nodes_torch_batched] = Vm_learnable
        return Vm_new

    def prepare_fixed_inputs(self):
        # convert parameters to torch
        pvs = np.concatenate([self.pv_nodes_solver + i * self.nb_active_buses for i in range(self.batch_size)])
        pqs = np.concatenate([self.pq_nodes_solver + i * self.nb_active_buses for i in range(self.batch_size)])

        self.pv_nodes_torch_batched = torch.tensor(pvs, requires_grad=False)
        self.pq_nodes_torch_batched = torch.tensor(pqs, requires_grad=False)

        # self.slack_ids_torch_batched = torch.tensor(self.slack_ids_solver[0])  # TODO not used right now

        values = torch.tensor(self.Ybus_solver.data, requires_grad=False)
        crow_indices = torch.tensor(self.Ybus_solver.indptr, requires_grad=False)
        col_indices = torch.tensor(self.Ybus_solver.indices, requires_grad=False)
        shape = self.Ybus_solver.shape

        # print("shape", shape)  # 118,118

        block_values = []  # simple concatenation

        block_crow_indices = []  # shifting with nnz, ignoring first range (always 0)
        # example of how this works:
        # indptr = [0, 2, 4] means row0 has values values[0:2], row1 has values values[2:4]
        # so if we want to create a blockdiagonal matrix, the new indptr is [0, 2, 4, 6, 8]
        # notice that every index is shifted by 4 (nnz) but the 0 is left out

        block_col_indices = []  # shifting by height
        block_shape = (shape[0] * self.batch_size, shape[1] * self.batch_size)

        nnz = values.numel()

        for i in range(self.batch_size):
            if i == 0:
                block_crow_indices.append(crow_indices)
            else:
                block_crow_indices.append(crow_indices[1:] + i * nnz)  # 0 2 4 --> 0 2 4 6 8
            block_col_indices.append(col_indices + i * self.nb_active_buses)
            block_values.append(values)

        block_values = torch.cat(block_values)  # flatten
        block_crow_indices = torch.cat(block_crow_indices)
        block_col_indices = torch.cat(block_col_indices)

        self.Ybus_torch_batched = torch.sparse_csr_tensor(block_crow_indices, block_col_indices, block_values,
                                                          block_shape, requires_grad=False, device=self.device)
        # sanity check
        #dense_ybus_batched = self.Ybus_torch_batched.to_dense()
        #assert torch.allclose(dense_ybus_batched[0:self.nb_active_buses, 0:self.nb_active_buses] ,
        #                     dense_ybus_batched[self.nb_active_buses:2*self.nb_active_buses, self.nb_active_buses:2*self.nb_active_buses])

        self.Va_fixed_batched = torch.tensor(np.angle(self.V), requires_grad=False, device=self.device)  # works for batched version as well
        self.Vm_fixed_batched = torch.tensor(np.abs(self.V), requires_grad=False, device=self.device)  #

    def run_pf(self):
        pass

    def run_time_series_batched(self, evaluate_losses=False):  # prod_p, prod_v, load_p and load_q not used
        self.prepare_fixed_inputs()
        loss_fn = self.hyperparams["loss_fn"]
        max_iter = self.hyperparams["max_iter"]
        tol = self.hyperparams["tol"]

        losses = []  # shape [num_time_steps, num_iterations]
        times = []

        Va_ = np.angle(self.V)  # elementwise
        Vm_ = np.abs(self.V)  # elementwise
        self.Va_learnable = torch.tensor(
            Va_[torch.concatenate([self.pv_nodes_torch_batched, self.pq_nodes_torch_batched])],
            requires_grad=True, device=self.device)

        self.Vm_learnable = torch.tensor(Vm_[self.pq_nodes_torch_batched], requires_grad=True, device=self.device)

        Sbus_real_torch = torch.tensor(np.real(self.Sbus_solver_batched), requires_grad=False)
        Sbus_imag_torch = torch.tensor(np.imag(self.Sbus_solver_batched), requires_grad=False)
        self.Sbus_torch_batched = torch.complex(Sbus_real_torch, Sbus_imag_torch)

        # transfer tensors to gpu
        if self.use_gpu:
            self.Sbus_torch_batched = self.Sbus_torch_batched.to(self.device)

        self.params = [self.Vm_learnable, self.Va_learnable]
        optimizer_kwargs = self.hyperparams["optimizer_kwargs"]
        self.optimizer = self.hyperparams["optimizer_class"](self.params, **optimizer_kwargs)

        scheduler_kwargs = self.hyperparams["scheduler_kwargs"]
        self.scheduler = self.hyperparams["scheduler_class"](self.optimizer, **scheduler_kwargs)

        individual_losses = []
        if evaluate_losses:
            individual_losses = np.zeros((self.batch_size, max_iter))

        start_time = time.perf_counter()
        for i in range(max_iter):
            time_stamp = time.perf_counter()
            times.append(time_stamp - start_time)

            self.optimizer.zero_grad()
            Vm_torch_batched = self.reconstruct_Vm(self.Vm_learnable)
            Va_torch_batched = self.reconstruct_Va(self.Va_learnable)
            V_torch_batched = Vm_torch_batched * torch.exp(1j * Va_torch_batched)

            # forward pass
            self.Ybus_conj_torch_batched = torch.conj(self.Ybus_torch_batched)
            V_conj_torch_batched = torch.conj(V_torch_batched)
            S_calc_torch_batched = V_torch_batched * torch.matmul(self.Ybus_conj_torch_batched, V_conj_torch_batched)

            # loss function
            S_calc_real_relevant_parts = S_calc_torch_batched.real[
                torch.concatenate([self.pv_nodes_torch_batched, self.pq_nodes_torch_batched])]
            S_calc_imag_relevant_parts = S_calc_torch_batched.imag[self.pq_nodes_torch_batched]
            out = torch.concatenate([S_calc_real_relevant_parts, S_calc_imag_relevant_parts])
            # target
            Sbus_real_relevant_parts = self.Sbus_torch_batched.real[
                torch.concatenate([self.pv_nodes_torch_batched, self.pq_nodes_torch_batched])]
            Sbus_imag_relevant_parts = self.Sbus_torch_batched.imag[self.pq_nodes_torch_batched]
            target = torch.concatenate([Sbus_real_relevant_parts, Sbus_imag_relevant_parts])

            loss = loss_fn(out, target)
            # TODO alternative: maybe train using shape (batchsize, num_active_buses) and really do independent training
            losses.append(loss.item())

            if evaluate_losses:
                chunk_size = out.shape[0] // self.batch_size
                for batch in range(self.batch_size):
                    local_out = out[batch * chunk_size:(batch + 1) * chunk_size]
                    local_target = target[batch * chunk_size:(batch + 1) * chunk_size]

                    local_loss = loss_fn(local_out, local_target)
                    individual_losses[batch, i] = local_loss.item()

            if loss < self.hyperparams["tol"]:
                print("converged to tolerance level")
            else:
                loss.backward()  # segfault after a few iterations... dafuq?? :D

                self.optimizer.step()

                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(loss.item())
                else:
                    if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                        self.scheduler.step(loss.item())
                    else:
                        self.scheduler.step()
        return np.array(losses), times, individual_losses

    def run_pf_super_grid(self, evaluate_losses, strategy, strategy_amount_param):  # prod_p, prod_v, load_p and load_q not used

        if strategy == "no_connections":
            # leave Ybus in a block diagonal structure without any connections
            pass
        elif strategy == "total_random":
            # adds batch_size * strategy_amount_param many random connections
            new_indices = get_k_random_zero_entries_without_diagonal(self.Ybus_solver, strategy_amount_param * self.batch_size)
            # [(i1,j1), (j1,i1), (i2,j2), (j2,i2), ...]
            new_values = get_k_random_values_duplicated(strategy_amount_param * self.batch_size)
            # TODO new_values is currently hardcoded

            rows, cols = zip(*new_indices)
            shape = self.Ybus_solver.shape
            update = coo_matrix((new_values, (rows, cols)), shape=shape).tocsr()

            Ybus_solver_new = self.Ybus_solver + update
            self.Ybus_solver = Ybus_solver_new

        elif strategy == "linear_random":
            pass
        elif strategy == "pairwise_random":
            pass
        else:
            pass

        self.prepare_fixed_inputs()

        loss_fn = self.hyperparams["loss_fn"]
        max_iter = self.hyperparams["max_iter"]
        tol = self.hyperparams["tol"]

        losses = []  # shape [num_time_steps, num_iterations]
        times = []

        Va_ = np.angle(self.V)  # elementwise
        Vm_ = np.abs(self.V)  # elementwise
        self.Va_learnable = torch.tensor(
            Va_[torch.concatenate([self.pv_nodes_torch_batched, self.pq_nodes_torch_batched])],
            requires_grad=True, device=self.device)

        self.Vm_learnable = torch.tensor(Vm_[self.pq_nodes_torch_batched], requires_grad=True, device=self.device)

        Sbus_real_torch = torch.tensor(np.real(self.Sbus_solver_batched), requires_grad=False)
        Sbus_imag_torch = torch.tensor(np.imag(self.Sbus_solver_batched), requires_grad=False)
        self.Sbus_torch_batched = torch.complex(Sbus_real_torch, Sbus_imag_torch)

        # transfer tensors to gpu
        if self.use_gpu:
            self.Sbus_torch_batched = self.Sbus_torch_batched.to(self.device)

        self.params = [self.Vm_learnable, self.Va_learnable]
        optimizer_kwargs = self.hyperparams["optimizer_kwargs"]
        self.optimizer = self.hyperparams["optimizer_class"](self.params, **optimizer_kwargs)

        scheduler_kwargs = self.hyperparams["scheduler_kwargs"]
        self.scheduler = self.hyperparams["scheduler_class"](self.optimizer, **scheduler_kwargs)

        individual_losses = []
        if evaluate_losses:
            individual_losses = np.zeros((self.batch_size, max_iter))

        start_time = time.perf_counter()
        for i in range(max_iter):
            time_stamp = time.perf_counter()
            times.append(time_stamp - start_time)

            self.optimizer.zero_grad()
            Vm_torch_batched = self.reconstruct_Vm(self.Vm_learnable)
            Va_torch_batched = self.reconstruct_Va(self.Va_learnable)
            V_torch_batched = Vm_torch_batched * torch.exp(1j * Va_torch_batched)

            # forward pass
            self.Ybus_conj_torch_batched = torch.conj(self.Ybus_torch_batched)
            V_conj_torch_batched = torch.conj(V_torch_batched)
            S_calc_torch_batched = V_torch_batched * torch.matmul(self.Ybus_conj_torch_batched, V_conj_torch_batched)

            # loss function
            S_calc_real_relevant_parts = S_calc_torch_batched.real[
                torch.concatenate([self.pv_nodes_torch_batched, self.pq_nodes_torch_batched])]
            S_calc_imag_relevant_parts = S_calc_torch_batched.imag[self.pq_nodes_torch_batched]
            out = torch.concatenate([S_calc_real_relevant_parts, S_calc_imag_relevant_parts])
            # target
            Sbus_real_relevant_parts = self.Sbus_torch_batched.real[
                torch.concatenate([self.pv_nodes_torch_batched, self.pq_nodes_torch_batched])]
            Sbus_imag_relevant_parts = self.Sbus_torch_batched.imag[self.pq_nodes_torch_batched]
            target = torch.concatenate([Sbus_real_relevant_parts, Sbus_imag_relevant_parts])

            loss = loss_fn(out, target)
            # TODO alternative: maybe train using shape (batchsize, num_active_buses) and really do independent training
            losses.append(loss.item())

            if evaluate_losses:
                chunk_size = out.shape[0] // self.batch_size
                for batch in range(self.batch_size):
                    local_out = out[batch * chunk_size:(batch + 1) * chunk_size]
                    local_target = target[batch * chunk_size:(batch + 1) * chunk_size]

                    local_loss = loss_fn(local_out, local_target)
                    individual_losses[batch, i] = local_loss.item()

            if loss < self.hyperparams["tol"]:
                print("converged to tolerance level")
            else:
                loss.backward()  # segfault after a few iterations... dafuq?? :D

                self.optimizer.step()

                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(loss.item())
                else:
                    if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                        self.scheduler.step(loss.item())
                    else:
                        self.scheduler.step()
        return np.array(losses), times, individual_losses
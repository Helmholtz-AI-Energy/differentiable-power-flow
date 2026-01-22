import time

import numpy as np
from scipy.sparse import coo_matrix
import torch
import copy

from dpf.solvers.abstract_powerflow_solver import AbstractPowerFlowSolver


def calculate_mismatch(Sbus, V, Ybus):
    # computes S_calc = V * np.conj(I) = V * np.conj(Ybus * V)
    tmp = Ybus * V
    tmp = np.conjugate(tmp)
    mis = V * tmp - Sbus
    return mis


def check_convergence(mismatch, pv_nodes, pq_nodes, tolerance_pu):
    # F consists of the active power mismatch of pv-nodes and active+reactive power mismatches of pq-nodes
    # so we take the real part for the active power and the imaginary part for the reactive power
    real_ = np.real(mismatch)
    imag_ = np.imag(mismatch)
    # F = np.concatenate([real_[slack_id_], real_[pv_nodes_], real_[pq_nodes_], imag_[pq_nodes_]])
    F = np.concatenate(
        [real_[pv_nodes], real_[pq_nodes], imag_[pq_nodes]]
    )  # slack is the first variable in BaseAlgo::evaluate_Fx for multiple slacks
    norm_inf = np.linalg.norm(F, np.inf)
    converged = norm_inf < tolerance_pu
    return converged, F


class TorchPowerFlowSolver(AbstractPowerFlowSolver):
    """
    Power flow solver using the Differentiable Simulation approach.
    """

    def __init__(self, backend, hyperparams=None):
        super().__init__(backend)

        self.device = None
        self.use_gpu = None
        self.normalized_mismatches = None
        self.eval_dict = None
        self.voltages = None
        self.gradients = None
        self.optimizer_init_time = None
        self.times_list = None
        self.inf_norm_loss_list = None
        self.Vm_fixed = None
        self.Va_fixed = None
        self.Sbus_torch = None
        self.Ybus_torch = None
        self.pv_nodes_torch = None
        self.pq_nodes_torch = None
        self.slack_id_torch = None
        self.loss_list = None
        self.best_loss = None
        self.best_checkpoint = None
        if hyperparams is None:
            # just one set of hyperparameters
            self.hyperparams = {
                "optimizer_class": torch.optim.RMSprop,
                "optimizer_kwargs": {
                    "lr": 0.00012283,
                    "alpha": 0.86468552,
                    "weight_decay": 0.15580241,
                    "momentum": 0.99858620,
                },
                "scheduler_class": torch.optim.lr_scheduler.ReduceLROnPlateau,
                "scheduler_kwargs": {
                    "factor": 0.4737,
                    "patience": 40,
                    "threshold_mode": "rel",
                    "threshold": 0.091412,
                    "cooldown": 80,
                },
                "loss_fn": torch.nn.MSELoss(),
                "max_iter": 1000,
                "tol": 1e-8,
            }
        else:
            self.hyperparams = hyperparams

    def set_gpu_usage(self, use_gpu=False):
        if use_gpu:
            assert torch.cuda.is_available()
        device = torch.device("cuda" if use_gpu else "cpu")
        self.use_gpu = use_gpu
        self.device = device

    def reconstruct_Va(self, Va_learnable):
        """
        A helping method that reconstructs the whole voltage angle vector from the fixed parts and learnable parts.
        :param Va_learnable: Learnable part of the voltage angle vector.
        :return: Full voltage angle vector.
        """
        Va_new = self.Va_fixed.clone()
        Va_new[torch.concatenate([self.pv_nodes_torch, self.pq_nodes_torch])] = (
            Va_learnable
        )
        return Va_new

    def reconstruct_Vm(self, Vm_learnable):
        """
        A helping method that reconstructs the whole voltage magnitude vector from the fixed and learnable parts.
        :param Vm_learnable: Learnable part of the voltage magnitude vector.
        :return: Full voltage magnitude vector.
        """
        Vm_new = self.Vm_fixed.clone()
        Vm_new[self.pq_nodes_torch] = Vm_learnable
        return Vm_new

    def add_new_random_connections_to_ybus(self, nb_new_random_connections, mean, std):
        if nb_new_random_connections == 0:
            return

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

        def get_k_random_values_duplicated(k, mean, std, seed=0):

            def random_complex(mean_, std_):
                sigma = std_ / np.sqrt(2)
                real_part = np.random.normal(loc=mean_.real, scale=sigma)
                imag_part = np.random.normal(loc=mean_.imag, scale=sigma)
                return real_part + 1j * imag_part

            # [z1,z1, z2,z2, ... , zk,zk]
            random_list = []
            for i in range(k):
                # get a random number
                z = (
                    0.00001515 + 0.00001515 * i
                )  # for now add a really smalll fixed number here
                # z = random_complex(mean, std)  # large std leads to larger values here. Solvability suffers from that
                random_list.append(z)
                random_list.append(z)

            return random_list

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

    def train(
        self,
        optimizer,
        params,
        scheduler,
        loss_fn,
        iterations,
        tol,
        checkpointing=False,
        save_gradients=False,
        save_voltages=False,
        report_metrics=False,
        save_normalized_mismatches=False,
    ):
        """
        Does the training steps of the differentiable simulation. "Training" here is used to calculate the solution
        voltage vector (and not to train learnable parameters to be used in inference later).
        """
        start = time.perf_counter()

        Vm_learnable = params[0]
        Va_learnable = params[1]

        loss_list = []
        times_list = []
        if checkpointing:
            best_checkpoint = [copy.deepcopy(Vm_learnable), copy.deepcopy(Va_learnable)]
        else:
            best_checkpoint = [Vm_learnable, Va_learnable]
        best_loss = torch.inf

        gradients = []
        voltages = []
        normalized_mismatches = []

        # evaluation lists
        mse_loss_list = []
        l1_loss_list = []
        linf_loss_list = []
        average_percentage_diff_list = []

        for i in range(iterations):

            time_stamp = time.perf_counter()
            times_list.append(time_stamp - start)

            optimizer.zero_grad()

            Vm_torch = self.reconstruct_Vm(Vm_learnable)
            Va_torch = self.reconstruct_Va(Va_learnable)
            V_torch = Vm_torch * torch.exp(1j * Va_torch)

            if save_voltages:
                voltages.append(
                    [copy.deepcopy(Vm_learnable), copy.deepcopy(Va_learnable)]
                )

            # forward pass
            Ybus_conj_torch = torch.conj(self.Ybus_torch)
            V_conj_torch = torch.conj(V_torch)
            S_calc_torch = V_torch * torch.matmul(Ybus_conj_torch, V_conj_torch)

            # loss function
            S_calc_real_relevant_parts = S_calc_torch.real[
                np.concatenate([self.pv_nodes_torch, self.pq_nodes_torch])
            ]
            S_calc_imag_relevant_parts = S_calc_torch.imag[self.pq_nodes_torch]
            out = torch.concatenate(
                [S_calc_real_relevant_parts, S_calc_imag_relevant_parts]
            )

            # target
            Sbus_real_relevant_parts = self.Sbus_torch.real[
                np.concatenate([self.pv_nodes_torch, self.pq_nodes_torch])
            ]
            Sbus_imag_relevant_parts = self.Sbus_torch.imag[self.pq_nodes_torch]
            target = torch.concatenate(
                [Sbus_real_relevant_parts, Sbus_imag_relevant_parts]
            )
            #
            # TODO maybe split up real and imaginary part of the loss to enable better scaling?
            loss = loss_fn(out, target)
            # inf_norm_loss = torch.max(torch.abs(target - out))
            # l1_loss_fn = torch.nn.L1Loss()
            # l1_loss = l1_loss_fn(out, target)
            # print("loss: mse/inf/l1", loss.item(), inf_norm_loss.item(), l1_loss.item())

            loss_list.append(loss.item())

            if report_metrics:
                mse_loss_fun = torch.nn.MSELoss()
                mse_loss = mse_loss_fun(out, target).detach().numpy()
                average_percentage_diff = (
                    torch.abs(out - target).sum() / torch.abs(target).sum()
                )
                average_percentage_diff = average_percentage_diff.detach().numpy()
                # print(average_percentage_diff * 100)

                # l1_loss_fn = torch.nn.L1Loss()
                # l1_loss = l1_loss_fn(out, target).detach().numpy()

                # linf_loss = torch.max(torch.abs(target - out)).detach().numpy()

                mse_loss_list.append(mse_loss)
                # l1_loss_list.append(l1_loss)
                # linf_loss_list.append(linf_loss)
                average_percentage_diff_list.append(average_percentage_diff)

            if loss.item() < best_loss:
                best_loss = loss.item()
                if checkpointing:
                    best_checkpoint = [
                        copy.deepcopy(Vm_learnable),
                        copy.deepcopy(Va_learnable),
                    ]

            if loss < self.hyperparams["tol"]:
                print("converged to tolerance level")
                if not checkpointing:
                    best_checkpoint = [Vm_learnable, Va_learnable]
                eval_dict = {
                    "mse": np.array(mse_loss_list),
                    "l1": np.array(l1_loss_list),
                    "linf": np.array(linf_loss_list),
                    "average_percentage_diff": np.array(average_percentage_diff_list),
                }
                return loss_list, best_checkpoint, best_loss, eval_dict

            loss.backward()

            if save_gradients:
                gradients.append(
                    [copy.deepcopy(Vm_learnable.grad), copy.deepcopy(Va_learnable.grad)]
                )

            if (
                save_normalized_mismatches
            ):  # TODO this only makes sense if target is not 0...
                normalized_mismatch = torch.abs(out - target) / (torch.abs(target))
                normalized_mismatches.append(normalized_mismatch)

            optimizer.step()

            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(loss.item())
            else:
                scheduler.step()

            if loss.item() < tol:
                break
        loss_list = np.array(loss_list)
        if save_gradients:
            self.gradients = gradients
        if save_voltages:
            self.voltages = voltages
        if save_normalized_mismatches:
            self.normalized_mismatches = normalized_mismatches

        if not checkpointing:
            best_checkpoint = [Vm_learnable, Va_learnable]

        eval_dict = {
            "mse": np.array(mse_loss_list),
            "l1": np.array(l1_loss_list),
            "linf": np.array(linf_loss_list),
            "average_percentage_diff": np.array(average_percentage_diff_list),
        }

        return loss_list, best_checkpoint, best_loss, times_list, eval_dict

    def prepare_non_learnable_inputs(self):
        """
        Converts fixed inputs from solver form to torch tensors.
        """
        # convert parameters to torch
        self.pv_nodes_torch = torch.tensor(self.pv_nodes_solver)
        self.pq_nodes_torch = torch.tensor(self.pq_nodes_solver)
        self.slack_id_torch = torch.tensor(self.slack_ids_solver[0])

        # Sbus, make sure Sbus is complex in torch
        Sbus_real_torch = torch.tensor(np.real(self.Sbus_solver), requires_grad=False)
        Sbus_imag_torch = torch.tensor(np.imag(self.Sbus_solver), requires_grad=False)
        self.Sbus_torch = torch.complex(Sbus_real_torch, Sbus_imag_torch).to(
            self.device
        )

        # Ybus, make sure Ybus is sparse and complex in pytorch
        # see https://github.com/pytorch/pytorch/issues/50690
        values = self.Ybus_solver.data
        crow_indices = self.Ybus_solver.indptr
        col_indices = self.Ybus_solver.indices
        shape = self.Ybus_solver.shape
        self.Ybus_torch = torch.sparse_csr_tensor(
            torch.tensor(crow_indices, dtype=torch.int64),
            torch.tensor(col_indices, dtype=torch.int64),
            torch.tensor(values, dtype=torch.complex128),
            shape,
            device=self.device,
        )

        self.Va_fixed = torch.tensor(
            np.angle(self.V), requires_grad=False, device=self.device
        )
        self.Vm_fixed = torch.tensor(
            np.abs(self.V), requires_grad=False, device=self.device
        )

    def run_pf(
        self,
        report_metrics=False,
        checkpointing=False,
        save_gradients=False,
        save_voltages=False,
        save_normalized_mismatches=False,
    ):
        """
        Power flow solution method. Inputs are prepared and the differentiable simulation method is used.
        """
        start_time = time.perf_counter()

        self.prepare_non_learnable_inputs()

        Va_ = np.angle(self.V)
        Vm_ = np.abs(self.V)

        # see https://ocw.tudelft.nl/wp-content/uploads/PowerFlow.pdf slide 18
        # slack: Vm/Va known , P/Q unknown
        # pv-nodes: P/Vm known, Va/Q unknown
        # pq-nodes: P/Q known, Vm/Va unknown
        # ---> Vm unknown for pq , Va unknown for pv/pq

        Va_learnable = torch.tensor(
            Va_[np.concatenate([self.pv_nodes_solver, self.pq_nodes_solver])],
            requires_grad=True,
            device=self.device,
        )
        Vm_learnable = torch.tensor(
            Vm_[self.pq_nodes_torch], requires_grad=True, device=self.device
        )

        params = [Vm_learnable, Va_learnable]

        optimizer_kwargs = self.hyperparams["optimizer_kwargs"]
        time_opt_before = time.perf_counter()
        optimizer = self.hyperparams["optimizer_class"](params, **optimizer_kwargs)
        time_opt_after = time.perf_counter()
        self.optimizer_init_time = time_opt_after - time_opt_before
        # print("optimizer init time", time_opt_after-time_opt_before)

        scheduler_kwargs = self.hyperparams["scheduler_kwargs"]
        scheduler = self.hyperparams["scheduler_class"](optimizer, **scheduler_kwargs)

        loss_fn = self.hyperparams["loss_fn"]
        max_iter = self.hyperparams["max_iter"]
        tol = self.hyperparams["tol"]

        time_before_train = time.perf_counter()
        # print("time before train: ", time_before_train-start_time)
        loss_list, best_checkpoint, best_loss, times_list, eval_dict = self.train(
            optimizer,
            params,
            scheduler,
            loss_fn,
            max_iter,
            tol,
            checkpointing=checkpointing,
            save_gradients=save_gradients,
            save_voltages=save_voltages,
            report_metrics=report_metrics,
            save_normalized_mismatches=save_normalized_mismatches,
        )

        #  update the solution
        self.Vm = self.reconstruct_Vm(best_checkpoint[0]).detach().cpu().numpy()
        self.Va = self.reconstruct_Va(best_checkpoint[1]).detach().cpu().numpy()
        self.V = self.Vm * np.exp(1j * self.Va)
        self.S_calc = self.V * np.conjugate(self.Ybus_solver * self.V)

        self.loss_list = loss_list
        self.times_list = [x + (time_before_train - start_time) for x in times_list]
        self.best_loss = best_loss
        self.best_checkpoint = best_checkpoint
        self.eval_dict = eval_dict

    def calc_magnitudes_with_given_voltage_magnitudes(self, Vm):
        v_or_ = np.zeros(len(self.line_status))
        v_ex_ = np.zeros(len(self.line_status))

        nb_lines = self.line_status.shape[0]

        for el_id in range(nb_lines):  # 186
            if not self.line_status[el_id]:
                continue

            bus_solver_id_or, bus_solver_id_ex = self.get_bus_solver_ids(el_id)

            v_or_[el_id] = (
                Vm[bus_solver_id_or] * self.bus_vn_kv[bus_solver_id_or]
            )  # in kV
            v_ex_[el_id] = Vm[bus_solver_id_ex] * self.bus_vn_kv[bus_solver_id_ex]

        return v_or_, v_ex_

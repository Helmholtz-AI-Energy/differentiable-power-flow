"""
This class is made for the purpose of handling time series where the grid stays the same but the injections change.
"""

import numpy as np
import torch

from dpf.solvers.abstract_powerflow_solver import AbstractPowerFlowSolver


class TimeSeriesPowerFlowSolver(AbstractPowerFlowSolver):
    """
    Differentiable Power Flow solver for time series. Multiple time steps are processed one by one
    (and not in a batched way).
    """

    def __init__(self, backend, hyperparams=None, continuation_hyperparams=None):
        super().__init__(backend)

        self.average_percentage_diffes = None
        self.Ybus_conj_torch = None
        self.scheduler = None
        self.optimizer = None
        self.params = None
        self.Vm_learnable = None
        self.Va_learnable = None
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
            self.hyperparams = {
                "optimizer_class": torch.optim.Adam,
                "optimizer_kwargs": {"lr": 0.003377, "betas": (0.979681, 0.963442)},
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
        else:
            self.hyperparams = hyperparams
        self.continuation_hyperparams = continuation_hyperparams

    def reconstruct_Va(self, Va_learnable):
        Va_new = self.Va_fixed.clone()
        Va_new[torch.concatenate([self.pv_nodes_torch, self.pq_nodes_torch])] = (
            Va_learnable
        )
        return Va_new

    def reconstruct_Vm(self, Vm_learnable):
        Vm_new = self.Vm_fixed.clone()
        Vm_new[self.pq_nodes_torch] = Vm_learnable
        return Vm_new

    def prepare_fixed_inputs(self):
        # convert parameters to torch
        self.pv_nodes_torch = torch.tensor(self.pv_nodes_solver)
        self.pq_nodes_torch = torch.tensor(self.pq_nodes_solver)
        self.slack_id_torch = torch.tensor(self.slack_ids_solver[0])

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
        )

        self.Va_fixed = torch.tensor(np.angle(self.V), requires_grad=False)
        self.Vm_fixed = torch.tensor(np.abs(self.V), requires_grad=False)

    def run_pf(self):
        pass

    def run_time_series(
        self,
        prod_p,
        prod_v,
        load_p,
        load_q,
        Sbuses,
        freeze_start_params=False,
        do_only_first_time_step=False,
        report_metrics=False,
    ):

        # productions and loads are ignored for power-flows since Sbus contains the relevant information already

        self.prepare_fixed_inputs()

        loss_fn = self.hyperparams["loss_fn"]
        start_iter = self.hyperparams["start_iter"]
        max_iter = self.continuation_hyperparams["max_iter"]
        tol = self.hyperparams["tol"]
        total_time_steps = Sbuses.shape[0]

        losses = []  # shape [num_time_steps, num_iterations]
        average_percentage_diffes = []

        for i in range(total_time_steps):
            # current_p = prod_p[i]
            # current_v = prod_v[i]  # what to do with this normally? ignore? it does not seem to have an effect..
            # current_load_p = load_p[i]
            # current_load_q = load_q[i]
            current_Sbus = Sbuses[i]

            # update changing values

            # this can take some time and is somewhat avoidable if we don't care about the backend correctness
            # post-processing might be affected
            # for power-flows this can be omitted here

            # TODO uncomment if necessary
            #  self.fill_backend_with_data(self.topo_vect, current_p, current_v, current_load_p, current_load_q)

            self.fillSbus(current_Sbus)  # sets self.Sbus_solver

            # now run power-flow by utilizing old solution and old fixed variables if existing

            if i == 0:
                Va_ = np.angle(self.V)
                Vm_ = np.abs(self.V)

                self.Va_learnable = torch.tensor(
                    Va_[np.concatenate([self.pv_nodes_solver, self.pq_nodes_solver])],
                    requires_grad=True,
                )
                self.Vm_learnable = torch.tensor(
                    Vm_[self.pq_nodes_torch], requires_grad=True
                )
                self.params = [self.Vm_learnable, self.Va_learnable]

                optimizer_kwargs = self.hyperparams["optimizer_kwargs"]
                self.optimizer = self.hyperparams["optimizer_class"](
                    self.params, **optimizer_kwargs
                )

                scheduler_kwargs = self.hyperparams["scheduler_kwargs"]
                self.scheduler = self.hyperparams["scheduler_class"](
                    self.optimizer, **scheduler_kwargs
                )
            else:
                if freeze_start_params is False:

                    # optimizer
                    # new optimizer object every time step
                    # optimizer_kwargs = self.continuation_hyperparams["optimizer_kwargs"]
                    # self.optimizer = self.continuation_hyperparams["optimizer_class"](self.params, **optimizer_kwargs)

                    # same optimizer object, overwrite parameters
                    for param_group in self.optimizer.param_groups:
                        # print(param_group['lr'])
                        # print(param_group['betas'])
                        param_group["lr"] = self.continuation_hyperparams[
                            "optimizer_kwargs"
                        ]["lr"]
                        param_group["betas"] = self.continuation_hyperparams[
                            "optimizer_kwargs"
                        ]["betas"]

                    # see https://pytorch.org/docs/stable/generated/torch.optim.Adam.html#torch.optim.Adam
                    # m_t and v_t (first moment and second moment) are stored as a state for Adam
                    # Insight: Momentum is bad for new solutions since they are located at different positions
                    # For Adam only:
                    for param in self.optimizer.state.values():
                        # print(param)
                        if "exp_avg" in param:  # Reset momentum buffer
                            param["exp_avg"].zero_()  # first moment

                        # TODO disable as well?
                        # if 'exp_avg_sq' in param: # Reset variance tracking
                        #   param['exp_avg_sq'].zero_()  # second moment

                    # scheduler
                    # do a completely new scheduler
                    scheduler_kwargs = self.continuation_hyperparams["scheduler_kwargs"]
                    self.scheduler = self.continuation_hyperparams["scheduler_class"](
                        self.optimizer, **scheduler_kwargs
                    )

            Sbus_real_torch = torch.tensor(
                np.real(self.Sbus_solver), requires_grad=False
            )
            Sbus_imag_torch = torch.tensor(
                np.imag(self.Sbus_solver), requires_grad=False
            )
            self.Sbus_torch = torch.complex(Sbus_real_torch, Sbus_imag_torch)

            # Vm_learnable = params[0]
            # Va_learnable = params[1]

            loss_list = []
            average_percentage_diff_list = []

            if i == 0:
                current_max_iter = start_iter
            else:
                current_max_iter = max_iter

            for k in range(current_max_iter):
                # print(i.k)

                self.optimizer.zero_grad()

                Vm_torch = self.reconstruct_Vm(self.Vm_learnable)
                Va_torch = self.reconstruct_Va(self.Va_learnable)
                V_torch = Vm_torch * torch.exp(1j * Va_torch)

                # forward pass
                if i == 0:
                    self.Ybus_conj_torch = torch.conj(self.Ybus_torch)
                V_conj_torch = torch.conj(V_torch)
                S_calc_torch = V_torch * torch.matmul(
                    self.Ybus_conj_torch, V_conj_torch
                )

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

                loss = loss_fn(out, target)
                loss_list.append(loss.item())

                if report_metrics:
                    average_percentage_diff = (
                        torch.abs(out - target).sum() / torch.abs(target).sum()
                    )
                    average_percentage_diff = average_percentage_diff.detach().numpy()
                    average_percentage_diff_list.append(average_percentage_diff)

                if loss < self.hyperparams["tol"]:
                    print("converged to tolerance level")
                else:
                    loss.backward()
                    self.optimizer.step()

                    if i == 0:
                        if isinstance(
                            self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau
                        ):
                            self.scheduler.step(loss.item())
                        else:
                            self.scheduler.step()

                    else:
                        if freeze_start_params:
                            pass
                        else:
                            if isinstance(
                                self.scheduler,
                                torch.optim.lr_scheduler.ReduceLROnPlateau,
                            ):
                                self.scheduler.step(loss.item())
                            else:
                                self.scheduler.step()

                #  loss_list = np.array(loss_list)
            losses.append(loss_list)
            average_percentage_diffes.append(average_percentage_diff_list)
            if do_only_first_time_step:
                break

        max_length = max(len(lst) for lst in losses)
        losses = [
            lst + [0] * (max_length - len(lst)) for lst in losses
        ]  # padding to uniform length
        losses = np.array(losses)

        if report_metrics:
            average_percentage_diffes = [
                lst + [0] * (max_length - len(lst)) for lst in average_percentage_diffes
            ]
            average_percentage_diffes = np.array(average_percentage_diffes)
            self.average_percentage_diffes = np.array(average_percentage_diffes)

        return losses

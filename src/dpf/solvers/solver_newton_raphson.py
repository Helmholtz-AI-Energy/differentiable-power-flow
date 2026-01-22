import numpy as np
from scipy.sparse import hstack, vstack
from scipy.sparse.linalg import (
    spsolve,
    lsqr,
    lsmr,
)  # for the linear equation system solved in newton raphson

from abstract_powerflow_solver import AbstractPowerFlowSolver


def calculate_mismatch(Sbus, V, Ybus):
    """

    :param Sbus: Power injection vector
    :param V: Voltage vector
    :param Ybus: Admittance matrix
    :return: mismatch Scalc-Sbus
    """
    # computes S_calc = V * np.conj(I) = V * np.conj(Ybus * V)
    tmp = Ybus * V
    tmp = np.conjugate(tmp)
    mis = V * tmp - Sbus
    return mis


def check_convergence(mismatch, pv_nodes, pq_nodes, tolerance_pu):
    """Checks convergence by calculating the infinity norm from the mismatch vector."""
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


def calculate_partial_derivatives(V_, Ybus_):
    """
    Calculates the partial derivatives used in the Jacobian matrix.
    :param V_: Voltage vector in solver form
    :param Ybus_: Admittance matrix in solver form
    :return: Partial derivatives dS_dVm and dS_dVa (magnitude and angle)
    """
    Vnorm = V_ / np.abs(V_)  # each complex vector has length one in Vnorm
    Ibus = Ybus_ * V_  # Ohms Law in Matrix Form, U=RI <-> I = U/R = U*Y
    conjIbus_Vnorm = np.conjugate(Ibus) * Vnorm  # I_conj * V_norm elementwise

    dS_dVm_ = Ybus_.copy()  # just init
    dS_dVa_ = Ybus_.copy()  # just init

    # The matrix Ybus_ is in compressed row storage format. Every column index exists but only some row indices are present.
    # Hence we iterate over all columns but only over existing rows.
    for col_id in range(V_.shape[0]):
        # iterate over non-zero elements in Ybus_
        start_ptr = Ybus_.indptr[col_id]
        end_ptr = Ybus_.indptr[col_id + 1]
        non_zero_row_indices = Ybus_.indices[start_ptr:end_ptr]

        for row_id in non_zero_row_indices:  # get only non-zero entries here
            el_ybus = Ybus_[row_id, col_id]

            dS_dVm_el = (
                np.conjugate(el_ybus * Vnorm[col_id]) * V_[row_id]
            )  # V_i Ybus_ij* (V_j*/|V_j|)
            dS_dVa_el = el_ybus * V_[col_id]  # ...

            if col_id == row_id:
                dS_dVm_el += conjIbus_Vnorm[row_id]
                dS_dVa_el -= Ibus[row_id]  # leads to  += I_i* j V_i

            my_i = 0 + 1j
            tmp_loop = my_i * V_[row_id]
            dS_dVa_el = (
                np.conjugate(-dS_dVa_el) * tmp_loop
            )  # ... -(Ybus_ij V_j)^* i V_i

            dS_dVm_[row_id, col_id] = dS_dVm_el
            dS_dVa_[row_id, col_id] = dS_dVa_el

    return dS_dVm_, dS_dVa_


def calculate_jacobian(V_, Ybus_, pvpq, pq_nodes_):
    """Calculates the Jacobian matrix similar to the c++ implementation in LightSim2Grid."""
    # See BaseNRAlgo<LinearSolver>::_dSbus_dV for jacobian calculation in file BaseNRAlgo.tpp

    dS_dVm_, dS_dVa_ = calculate_partial_derivatives(V_, Ybus_)

    # create Jacobian matrix J_ with ds_dVm_ and ds_dVa_
    """
    J has the shape
    | J11 | J12 |               | (pvpq, pvpq) | (pvpq, pq) |
    | --------- | = dimensions: | ------------------------- |
    | J21 | J22 |               |  (pq, pvpq)  | (pq, pq) |
    python implementation:
    J11 = dS_dVa[array([pvpq]).T, pvpq].real
    J12 = dS_dVm[array([pvpq]).T, pq].real
    J21 = dS_dVa[array([pq]).T, pvpq].imag
    J22 = dS_dVm[array([pq]).T, pq].imag
    """
    size_j = pvpq.shape[0] + pq_nodes_.shape[0]
    J_ = np.zeros((size_j, size_j))
    dS_dVa_r = np.real(dS_dVa_)
    dS_dVa_i = np.imag(dS_dVa_)
    dS_dVm_r = np.real(dS_dVm_)
    dS_dVm_i = np.imag(dS_dVm_)

    J11 = dS_dVa_r[
        np.ix_(pvpq.T, pvpq)
    ]  # use np.ix_ to extract a submatrix with given indices (instead of fancy indexing)
    J12 = dS_dVm_r[np.ix_(pvpq.T, pq_nodes_)]
    J21 = dS_dVa_i[np.ix_(pq_nodes_.T, pvpq)]
    J22 = dS_dVm_i[np.ix_(pq_nodes_.T, pq_nodes_)]

    # print(J11.shape, J12.shape, J21.shape, J22.shape)
    J_ = vstack(((hstack((J11, J12))), (hstack((J21, J22)))))
    return J_


class NRPowerFlowSolver(AbstractPowerFlowSolver):
    """Python implementation of the Newton-Raphson power-flow solver. This is purely educational as the C++
    implementation is faster."""

    def __init__(self, backend):

        super().__init__(backend)

    def run_pf(self):
        # inputs
        Ybus_ = self.Ybus_solver
        V_ = self.V
        Sbus_ = self.Sbus_solver
        slack_ids_solver_ = self.slack_ids_solver
        slack_weights_ = self.slack_weights_solver
        pv_nodes_ = self.pv_nodes_solver
        pq_nodes_ = self.pq_nodes_solver
        max_iteration_ = self.max_iteration_solver
        # max_iteration_ = 1
        tolerance_pu_ = self.tolerance_pu_solver

        # Goal: Compute voltage angles and magnitudes at each bus such that the mismatch is minimal.

        # create inverse arrays
        pvpq = np.concatenate([pv_nodes_, pq_nodes_])
        pvpq_inv = np.full((V_.shape), -1)
        pq_inv = np.full((V_.shape), -1)

        for inv_idx in range(pvpq.shape[0]):
            pvpq_inv[pvpq[inv_idx]] = inv_idx

        for inv_idx in range(pq_nodes_.shape[0]):
            pq_inv[pq_nodes_[inv_idx]] = inv_idx

        # current magnitude/angle
        Vm_ = np.abs(V_)
        Va_ = np.angle(V_)  # in radians

        # check if done already by computing the mismatch
        mis = calculate_mismatch(Sbus_, V_, Ybus_)
        converged, F = check_convergence(mis, pv_nodes_, pq_nodes_, tolerance_pu_)
        # print("converged before powerflow?: ", converged)

        for iteration in range(max_iteration_):
            # print("iteration: ", iteration)

            J_ = calculate_jacobian(V_, Ybus_, pvpq, pq_nodes_)
            self.J = J_
            # print("Jacobian calculated")

            # TODO factorization, maybe ignore this for now
            # use Linear Solver to solve with mismatch information and get new V_a and V_m
            # solves J (delta_x) = F  with the Jacobian J, mismatch F and
            # solves by factorizing J = LU

            dx = spsolve(J_, F)  # here just use any sparse solver to see if it works
            # dx, istop, itn, r1norm = lsqr(J_, F, iter_lim=10)[:4] # alternative: use least square method
            # dx = lsmr(J_, F)[0]
            # print("linear system solved")
            # print("istop, itn, r1norm", istop, itn, r1norm)

            # obtain V_a and V_m using the solution of the linear equation system
            if pv_nodes_.shape[0] > 0:
                Va_[pv_nodes_] -= dx[: pv_nodes_.shape[0]]

            if pq_nodes_.shape[0] > 0:
                Va_[pq_nodes_] -= dx[
                    pv_nodes_.shape[0] : pv_nodes_.shape[0] + pq_nodes_.shape[0]
                ]
                Vm_[pq_nodes_] -= dx[
                    pq_nodes_.shape[0]
                    + pv_nodes_.shape[0] : pq_nodes_.shape[0]
                    + pv_nodes_.shape[0]
                    + pq_nodes_.shape[0]
                ]

            # Reconstruct V_ using Vm_ and Va_

            # V_ = Vm_ * (np.cos(Va_) + my_i * np.sin(Va_)) # equivalent
            V_ = Vm_ * np.exp(1j * Va_)
            self.Vm = Vm_
            self.Va = Va_
            self.V = V_

            # check convergence
            mis = calculate_mismatch(Sbus_, V_, Ybus_)
            converged, F = check_convergence(mis, pv_nodes_, pq_nodes_, tolerance_pu_)

            if converged:
                # print(f"convergence reached after {iteration} steps")
                self.S_calc = V_ * np.conjugate(Ybus_ * V_)
                break

        # some optimizations that we do NOT do here but are present in the c++ code in LightSim2Grid:
        # save sparsity pattern of Jacobian and reuse it for other iterations
        # Use LR decomposition of Jacobian to solve the lineat set of equations
        # for jacobian calculation: Reuse similar parts where possible
        return

from abc import abstractmethod, ABC
import numpy as np
from grid2op.Action._BackendAction import _BackendAction
from grid2op.Action import CompleteAction
from scipy.sparse import csr_matrix
import copy


class AbstractPowerFlowSolver(ABC):

    def __init__(self, backend):

        self.backend = backend.copy()  # the original backend is not modified
        self.grid = self.backend._grid
        self.topo_vect = None

        # grid data
        self.generators = None
        self.generator_slack_id = None
        self.trafos = None
        self.power_lines = None

        # input data in solver form (only active buses considered)
        self.tolerance_pu_solver = 1e-8
        self.max_iteration_solver = 1000
        self.init_vm_pu_solver = None
        self.sn_mva_solver = None
        self.bus_vn_kv = None
        self.nb_buses = None
        self.nb_active_buses = None

        self.bus_statuses = None  # a bus is active or not, do not mistake this for line status
        self.id_solver_to_me = None  # length: number of active buses, gives global bus index
        self.id_me_to_solver = None  # length: number of all buses, gives solver
        self.slack_ids_solver = None  # [slack_id]
        self.slack_weights_solver = None
        self.pv_nodes_solver = None
        self.pq_nodes_solver = None
        self.Ybus_solver = None
        self.Sbus_solver = None

        # powerflow output data
        self.V = None
        self.Vm = None  # redundant, can be retrieved from V
        self.Va = None  # redundant, can be retrieved from V
        self.J = None  # Jacobian from the Newton-Raphson algorithm
        self.S_calc = None  # redundant, can be calculated from V and Ybus

        self.line_ex_to_topo_pos = None
        self.line_ex_to_subid = None
        self.line_or_to_topo_pos = None
        self.line_or_to_subid = None
        self.line_status = None

        # target value data in global form (all buses/lines, not just the active ones)
        self.a_or = None
        self.a_ex = None
        self.v_or = None
        self.v_ex = None
        self.theta_or = None
        self.theta_ex = None
        self.p_or = None
        self.p_ex = None
        self.q_or = None
        self.q_ex = None

        # TBD

    def preprocess(self, topo_vect, prod_p, prod_v, load_p, load_q, Ybus, Sbus, pv, line_status=None):
        # grid.pre_process_solver() # unfortunately we have no access to this method that is called during ac_pf
        # so we have to reimplement it
        self.fill_backend_with_data(topo_vect, prod_p, prod_v, load_p, load_q)
        if line_status is not None:
            self.line_status = line_status
        self.fetch_grid_data()
        self.find_active_buses()
        self.init_slack()
        self.fillYbus(Ybus)
        self.fillSbus(Sbus)
        self.fill_pv_pq(pv)
        # generators_.init_q_vector ??? set limits on generations?
        # dc_lines_.init_q_vector ?? set limits on dclines?
        #self.init_v(init_strategy)
        # generators_.set_vm(V, id_me_to_solver); ???
        # dc_lines_.set_vm(V, id_me_to_solver); ???

    def fill_backend_with_data(self, topo_vect, prod_p, prod_v, load_p, load_q):
        # see lips dcApproximationAS.py
        _bk_act_class = _BackendAction.init_grid(self.backend)
        _act_class = CompleteAction.init_grid(self.backend)
        modifer = _bk_act_class()
        act = _act_class()
        act.update({"set_bus": topo_vect,
                    "injection": {
                        "prod_p": prod_p,
                        "prod_v": prod_v,
                        "load_p": load_p,
                        "load_q": load_q,
                    }
                    })
        modifer += act
        self.backend.apply_action(modifer)
        self.topo_vect = topo_vect

    def fetch_grid_data(self):
        self.generators = self.grid.get_generators()
        self.power_lines = self.grid.get_lines()
        self.trafos = self.grid.get_trafos()

        self.bus_statuses = self.grid.get_bus_status()
        self.bus_vn_kv = self.grid.get_bus_vn_kv()
        self.nb_buses = len(self.bus_vn_kv)
        self.sn_mva_solver = self.grid.get_sn_mva()
        self.init_vm_pu_solver = self.grid.get_init_vm_pu()
        if self.line_status is None:
            self.line_status = self.backend.get_line_status()

        self.line_or_to_subid = self.backend.line_or_to_subid
        self.line_or_to_topo_pos = self.backend.line_or_pos_topo_vect
        self.line_ex_to_subid = self.backend.line_ex_to_subid
        self.line_ex_to_topo_pos = self.backend.line_ex_pos_topo_vect

    def find_active_buses(self):
        # see GridModel::init_Ybus()
        nb_buses = len(self.bus_vn_kv)
        id_me_to_solver = [-1 for i in range(
            nb_buses)]  # be default deactivated, active buses will have the active bus id (=solver bus id)
        id_solver_to_me = []

        active_bus_id = 0
        for bus_id in range(nb_buses):
            if self.bus_statuses[bus_id]:  # bus is active
                id_me_to_solver[bus_id] = active_bus_id
                (id_solver_to_me.append(bus_id))
                active_bus_id = active_bus_id + 1
        self.id_me_to_solver = id_me_to_solver
        self.id_solver_to_me = id_solver_to_me
        self.nb_active_buses = len(id_solver_to_me)

    def init_slack(self):
        # slack id single bus, not required to do
        for i, generator in enumerate(self.generators):
            if generator.is_slack:
                slack_id_global = generator.bus_id  # global bus id of the slack
                self.generator_slack_id = i
                self.slack_ids_solver = [self.id_me_to_solver[slack_id_global]]
                break  # assumes single slack

        self.slack_weights_solver = np.zeros(self.nb_active_buses)
        self.slack_weights_solver[self.slack_ids_solver] = 1.0  # slack weight entry of slack is 1, rest 0

    def fillYbus(self, Ybus=None):
        if Ybus is not None:
            # relabel to active buses!
            # create sparse csc matrix of shape (self.nb_active_buses, self.nb_active_buses)
            # this is the inverse of the get_Ybus() method in Gridmodel.h

            res = csr_matrix((self.nb_active_buses, self.nb_active_buses), dtype=np.complex128)

            Ybus_reshaped = csr_matrix(Ybus.reshape((self.nb_buses, self.nb_buses)))
            for col_id in range(self.nb_buses):
                col_id_solver = self.id_me_to_solver[col_id]
                if col_id_solver == -1:
                    continue

                # iterate over non-zero elements in Ybus_
                start_ptr = Ybus_reshaped.indptr[col_id]
                end_ptr = Ybus_reshaped.indptr[col_id + 1]
                non_zero_row_indices = Ybus_reshaped.indices[start_ptr:end_ptr]
                for row_id in non_zero_row_indices:
                    row_id_solver = self.id_me_to_solver[row_id]
                    if row_id_solver == -1:
                        continue
                    res[row_id_solver, col_id_solver] = Ybus_reshaped[row_id, col_id]
                self.Ybus_solver = res
        pass

    def fillSbus(self, Sbus=None):
        if Sbus is not None:
            res = np.ndarray(self.nb_active_buses, dtype=np.complex128)
            for i in range(self.nb_active_buses):
                global_bus_id = self.id_solver_to_me[i]
                res[i] = Sbus[global_bus_id]
            self.Sbus_solver = res
        pass

    def fill_pv_pq(self, pv=None):
        # analog to the fillpv_pq method in Gridmodel.cpp
        if pv is None:
            print("no pv provided")
            return

        res_pv = []
        res_pq = []

        for global_bus_id, is_pv in enumerate(pv):
            # only consider this if activated
            if not self.bus_statuses[global_bus_id]:
                continue
            bus_id_solver = self.id_me_to_solver[global_bus_id]
            if bus_id_solver == self.slack_ids_solver[0]:
                #print("slack found and eliminated in pvpq")
                continue
            if is_pv:
                res_pv.append(bus_id_solver)
            else:
                res_pq.append(bus_id_solver)

        # https://stackoverflow.com/questions/39325930/numpy-ndarray-with-more-that-32-dimensions
        # just numpy things... this gets me an ndarray from a list larger than 32....
        self.pv_nodes_solver = np.array(np.array(res_pv))
        self.pq_nodes_solver = np.array(np.array(res_pq))

    def init_v(self, strategy="dc", random_init_seed=0):
        if strategy == "ones":
            self.V = np.array(np.ones(self.nb_active_buses, dtype=np.complex128) * self.init_vm_pu_solver)
        if strategy == "uniform_complex":  # does not work well
            np.random.seed(random_init_seed)
            self.V = (np.random.uniform(-1, 1, self.nb_active_buses) +
                      1.j * np.random.uniform(-1, 1, self.nb_active_buses))
        if strategy == "dc":
            # using backend here to do dc power flow, this can also be done in python if wanted

            backend_copy = self.backend.copy()
            grid_copy = backend_copy._grid

            backend_copy.V = np.ones(backend_copy.nb_bus_total, dtype=np.complex_) * self.init_vm_pu_solver
            grid_copy.deactivate_result_computation()
            backend_copy.V[:] = 1.
            backend_copy._debug_Vdc = grid_copy.dc_pf(copy.deepcopy(backend_copy.V),
                                                      self.max_iteration_solver,
                                                      self.tolerance_pu_solver)
            grid_copy.reactivate_result_computation()
            V_init_global_bus = 1. * backend_copy._debug_Vdc
            # convert to solver bus ids
            V_init_solver = np.zeros(self.nb_active_buses, dtype=np.complex128)
            for bus_solver_id in range(self.nb_active_buses):
                global_bus_id = self.id_solver_to_me[bus_solver_id]
                tmp = V_init_global_bus[global_bus_id]
                V_init_solver[bus_solver_id] = tmp
            self.V = V_init_solver
        return

    @abstractmethod
    def run_pf(self):
        return None

    def calculate_l2_loss(self):
        """
        Uses the current voltage vector self.V to evaluate the power mismatch with the l2 loss.
        :return: l2 loss
        """
        Sbus = self.Sbus_solver
        V = self.V
        Ybus = self.Ybus_solver
        pv_nodes = self.pv_nodes_solver
        pq_nodes = self.pq_nodes_solver

        S_calc = V * np.conjugate(Ybus * V)
        pvpq = np.concatenate([pv_nodes, pq_nodes]).astype(int)

        S_calc_real_relevant_parts = np.real(S_calc)[pvpq]
        S_calc_imag_relevant_parts = np.imag(S_calc)[pq_nodes]
        out = np.concatenate([S_calc_real_relevant_parts, S_calc_imag_relevant_parts])

        Sbus_real_relevant_parts = np.real(Sbus)[pvpq]
        Sbus_imag_relevant_parts = np.imag(Sbus)[pq_nodes]
        target = np.concatenate([Sbus_real_relevant_parts, Sbus_imag_relevant_parts])

        mse = ((out - target) ** 2).mean(axis=0)

        return mse

    def calculate_average_percentage_diff(self):
        Sbus = self.Sbus_solver
        V = self.V
        Ybus = self.Ybus_solver
        pv_nodes = self.pv_nodes_solver
        pq_nodes = self.pq_nodes_solver

        S_calc = V * np.conjugate(Ybus * V)
        pvpq = np.concatenate([pv_nodes, pq_nodes]).astype(int)

        S_calc_real_relevant_parts = np.real(S_calc)[pvpq]
        S_calc_imag_relevant_parts = np.imag(S_calc)[pq_nodes]
        out = np.concatenate([S_calc_real_relevant_parts, S_calc_imag_relevant_parts])

        Sbus_real_relevant_parts = np.real(Sbus)[pvpq]
        Sbus_imag_relevant_parts = np.imag(Sbus)[pq_nodes]
        target = np.concatenate([Sbus_real_relevant_parts, Sbus_imag_relevant_parts])

        average_percentage_diff = np.abs(out - target).sum() / np.abs(target).sum()
        return average_percentage_diff


    def calc_ybus_individual_contributions(self):
        # calculate individual line contributions here. Ybus is an aggregation that contains these as well.
        # You cannot retrieve these four quantitites from the Ybus matrix directly.
        # The first part is calculated for powerlines and the second part for transformers
        # (See LineContainer.cpp-->_update_model_coeffs and TrafoContainer.cpp-->_update_model_coeffs
        # Although the code calculates both seperately,
        # the calculations afterwards for currents/powers/voltages is similar. Hence we combine both here.

        nb_powerlines = len(self.power_lines)
        nb_trafos = len(self.trafos)
        my_i = 0 + 1j

        yac_ff_ = np.zeros(nb_powerlines + nb_trafos, dtype=complex)
        yac_tt_ = np.zeros(nb_powerlines + nb_trafos, dtype=complex)
        yac_ft_ = np.zeros(nb_powerlines + nb_trafos, dtype=complex)
        yac_tf_ = np.zeros(nb_powerlines + nb_trafos, dtype=complex)

        for i in range(nb_powerlines):
            line = self.power_lines[i]
            r_pu = line.r_pu
            x_pu = line.x_pu
            h_or_pu = line.h_or_pu
            h_ex_pu = line.h_ex_pu
            #  bus_or_id = line.bus_or_id
            #  bus_ex_id = line.bus_ex_id

            ys = 1.0 / (r_pu + my_i * x_pu)
            h_or = my_i * h_or_pu
            h_ex = my_i * h_ex_pu
            yac_ff_[i] = (ys + h_or)
            yac_tt_[i] = (ys + h_ex)
            yac_tf_[i] = -ys
            yac_ft_[i] = -ys

        # the direction orex corresponds to hvlv and
        # the direction exor to lvhv (--> high voltage is origin, low voltage extremity)
        for i in range(nb_trafos):
            line = self.trafos[i]

            r_pu = line.r_pu
            x_pu = line.x_pu
            h_i = line.h_pu
            tau = line.ratio
            #  bus_or_id = line.bus_hv_id  # origin corresponds to high voltage
            #  bus_ex_id = line.bus_lv_id  # extremity corresponds to low voltage
            is_tap_hv_side = line.is_tap_hv_side
            theta_shift = line.shift_rad

            ys = 1.0 / (r_pu + my_i * x_pu)
            h = my_i * h_i * 0.5
            if not is_tap_hv_side:
                tau = 1.0 / tau

            eitheta_shift = 1 + 0j
            emitheta_shift = 1 + 0j
            if theta_shift != 0.0:
                eitheta_shift = np.cos(theta_shift) + 1j * np.sin(theta_shift)
                emitheta_shift = np.cos(theta_shift) - 1j * np.sin(theta_shift)

            yac_ff_[nb_powerlines + i] = (ys + h) / (
                    tau * tau)  # see https://matpower.org/docs/MATPOWER-manual.pdf formula 3.2
            yac_tt_[nb_powerlines + i] = (ys + h)
            yac_tf_[nb_powerlines + i] = -ys / tau * emitheta_shift
            yac_ft_[nb_powerlines + i] = -ys / tau * eitheta_shift
        return yac_ff_, yac_tt_, yac_ft_, yac_tf_

    def get_bus_solver_ids(self, global_line_id):
        n_sub = int(self.nb_buses / 2)
        topo_pos_or = self.line_or_to_topo_pos[global_line_id]
        topo_pos_ex = self.line_ex_to_topo_pos[global_line_id]
        sub_id_or = self.line_or_to_subid[global_line_id]
        sub_id_ex = self.line_ex_to_subid[global_line_id]
        local_bus_num_or = self.topo_vect[topo_pos_or]
        local_bus_num_ex = self.topo_vect[topo_pos_ex]
        global_bus_num_or = sub_id_or + (int(local_bus_num_or) - 1) * n_sub
        global_bus_num_ex = sub_id_ex + (int(local_bus_num_ex) - 1) * n_sub
        bus_solver_id_or = self.id_me_to_solver[global_bus_num_or]
        bus_solver_id_ex = self.id_me_to_solver[global_bus_num_ex]
        return bus_solver_id_or, bus_solver_id_ex

    def calc_thetas(self):
        theta_or_ = np.zeros(len(self.line_status))
        theta_ex_ = np.zeros(len(self.line_status))

        nb_lines = self.line_status.shape[0]

        for el_id in range(nb_lines):  # 186
            if not self.line_status[el_id]:
                continue

            # getting corresponding solver id,
            # similar to GenericContainer.cpp ->v_kv_from_vpu() or in LineContainer.cpp --> compute_results()
            bus_solver_id_or, bus_solver_id_ex = self.get_bus_solver_ids(el_id)

            # assigning thetas (voltage angles)
            theta_or_[el_id] = self.Va[bus_solver_id_or] * 180. / np.pi  # in degree
            theta_ex_[el_id] = self.Va[bus_solver_id_ex] * 180. / np.pi

        self.theta_or = theta_or_
        self.theta_ex = theta_ex_

    def calc_magnitudes(self):
        v_or_ = np.zeros(len(self.line_status))
        v_ex_ = np.zeros(len(self.line_status))

        nb_lines = self.line_status.shape[0]

        for el_id in range(nb_lines):  # 186
            if not self.line_status[el_id]:
                continue

            bus_solver_id_or, bus_solver_id_ex = self.get_bus_solver_ids(el_id)

            v_or_[el_id] = self.Vm[bus_solver_id_or] * self.bus_vn_kv[bus_solver_id_or]  # in kV
            v_ex_[el_id] = self.Vm[bus_solver_id_ex] * self.bus_vn_kv[bus_solver_id_ex]

        self.v_or = v_or_
        self.v_ex = v_ex_

    def calc_powers(self, yac_ff_, yac_tt_, yac_ft_, yac_tf_):
        p_or_ = np.zeros(len(self.line_status))
        p_ex_ = np.zeros(len(self.line_status))
        q_or_ = np.zeros(len(self.line_status))
        q_ex_ = np.zeros(len(self.line_status))

        nb_lines = self.line_status.shape[0]

        for el_id in range(nb_lines):  # 186
            if not self.line_status[el_id]:
                continue

            bus_solver_id_or, bus_solver_id_ex = self.get_bus_solver_ids(el_id)

            # see LineContainer.cpp --> compute_results. We have to calculate yac_ff, yac_tf, yac_ft and yac_tt first
            # for the calculation, see _update_model_coeffs().
            Eor = self.V[bus_solver_id_or]
            Eex = self.V[bus_solver_id_ex]

            I_orex = yac_ff_[el_id] * Eor + yac_ft_[el_id] * Eex
            I_exor = yac_tt_[el_id] * Eex + yac_tf_[el_id] * Eor
            S_orex = Eor * np.conjugate(I_orex)
            S_exor = Eex * np.conjugate(I_exor)

            # assigning active powers
            p_or_[el_id] = np.real(S_orex) * self.sn_mva_solver  # in MW
            p_ex_[el_id] = np.real(S_exor) * self.sn_mva_solver
            q_or_[el_id] = np.imag(S_orex) * self.sn_mva_solver  # in MVar
            q_ex_[el_id] = np.imag(S_exor) * self.sn_mva_solver

        self.p_or = p_or_
        self.p_ex = p_ex_
        self.q_or = q_or_
        self.q_ex = q_ex_

    def calc_currents(self):
        # see GenericContainer::_get_amps to calc the currents
        p2q2_or_ = np.sqrt(np.array(self.p_or) * np.array(self.p_or) + np.array(self.q_or) * np.array(self.q_or))
        p2q2_ex_ = np.sqrt(np.array(self.p_ex) * np.array(self.p_ex) + np.array(self.q_ex) * np.array(self.q_ex))
        v_tmp_or = self.v_or.copy()
        v_tmp_ex = self.v_ex.copy()
        for i, v in enumerate(v_tmp_or):
            if v == 0.0:
                v_tmp_or[i] = 1.0
        for i, v in enumerate(v_tmp_ex):
            if v == 0.0:
                v_tmp_ex[i] = 1.0

        _1_sqrt_3 = 1.0 / np.sqrt(3.0)
        a_or_ = p2q2_or_ * _1_sqrt_3 / v_tmp_or
        self.a_or = a_or_ * 1000  # convert to A from kA, see lightSimBackend --> runpf
        a_ex_ = p2q2_ex_ * _1_sqrt_3 / v_tmp_ex
        self.a_ex = a_ex_ * 1000

    def post_process(self):
        # calculate the currents and powers acting upon the transmission lines

        self.calc_thetas()
        self.calc_magnitudes()
        yac_ff_, yac_tt_, yac_ft_, yac_tf_ = self.calc_ybus_individual_contributions()
        self.calc_powers(yac_ff_, yac_tt_, yac_ft_, yac_tf_)
        self.calc_currents()

    def extract_results(self):
        return (self.theta_or, self.theta_ex, self.v_or, self.v_ex,
                self.a_or, self.a_ex, self.p_or, self.p_ex, self.q_or, self.q_ex)

from dpf.scripts.ex1_running_torch_solver import main as ex1_main
from dpf.scripts.ex3_pareto_plot import main as ex3_main
from dpf.scripts.ex3_pareto_plot_eval import main as ex3_main_eval
from dpf.scripts.ex4_data_generation import main as ex4_data
from dpf.scripts.ex4a_scaling import main as ex4a_main
from dpf.scripts.ex4a_scaling_eval import main as ex4a_main_eval
from dpf.scripts.ex4b_scaling import main as ex4b_main
from dpf.scripts.ex4b_scaling_eval import main as ex4b_main_eval
from dpf.scripts.ex5_data_generation import main as ex5_data
from dpf.scripts.ex5a_time_series import main as ex5a_main
from dpf.scripts.ex5a_time_series_eval import main as ex5a_main_eval
from dpf.scripts.ex5b_time_series_batching import main as ex5b_main
from dpf.scripts.ex5b_time_series_batching_eval import main as ex5b_main_eval
from dpf.scripts.ex5c_time_series_batching_pegase9241 import main as ex5c_main
from dpf.scripts.ex5c_time_series_batching_pegase9241_eval import main as ex5c_main_eval
from dpf.scripts.ex6a_time_series_hypergrid_NR import main as ex6a_main
from dpf.scripts.ex6a_time_series_hypergrid_NR_eval import main as ex6a_main_eval
from dpf.scripts.ex6b_time_series_hypergrid_DPF import main as ex6b_main
from dpf.scripts.ex6b_time_series_hypergrid_DPF_eval import main as ex6b_main_eval
from dpf.scripts.ex6c_evaluate_ab import main as ex6c_main
from dpf.scripts.ex7a_single_grid_sparse_vs_dense_NR import main as ex7a_main
from dpf.scripts.ex7a_single_grid_sparse_vs_dense_NR_eval import main as ex7a_main_eval
from dpf.scripts.ex7b_single_grid_sparse_vs_dense_DPF import main as ex7b_main
from dpf.scripts.ex7b_single_grid_sparse_vs_dense_DPF_eval import main as ex7b_main_eval
from dpf.scripts.ex7c_evaluate import main as ex7c_main
from dpf.scripts.ex9a_super_grid_NR import main as ex9a_main
from dpf.scripts.ex9a_super_grid_NR_eval import main as ex9a_main_eval
from dpf.scripts.ex9b_hypergrid_DPF import main as ex9b_main
from dpf.scripts.ex9b_hypergrid_DPF_eval import main as ex9b_main_eval
from dpf.scripts.ex9c_eval import main as ex9c_main
from dpf.scripts.ex10a_wall_clock_scaling import main as ex10a_main
from dpf.scripts.ex10a_wall_clock_scaling_eval import main as ex10a_main_eval


def main():
    ex1_main()

    ex3_main()
    ex3_main_eval()

    ex4_data()
    ex4a_main()
    ex4a_main_eval()
    ex4b_main()
    ex4b_main_eval()

    ex5_data()
    ex5a_main()
    ex5a_main_eval()
    ex5b_main()
    ex5b_main_eval()
    ex5c_main()
    ex5c_main_eval()

    ex6a_main()
    ex6a_main_eval()
    ex6b_main()
    ex6b_main_eval()
    ex6c_main()

    ex7a_main()
    ex7a_main_eval()
    ex7b_main()
    ex7b_main_eval()
    ex7c_main()

    ex9a_main()
    ex9a_main_eval()
    ex9b_main()
    ex9b_main_eval()
    ex9c_main()

    ex10a_main()
    ex10a_main_eval()


if __name__ == "__main__":
    main()

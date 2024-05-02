# environment parameters
env = 'test'
THREADS = 12

# run parameters
N = 100 if env == 'test' else 100 # 100, 200, 300, 400, 500, 1000
G = 2500 if env == 'test' else 10000
NR = 100 if env == 'test' else 100

# convergence parameters
EPS = 0.0001
N_LAST_GENS = 10
DELTA = SIGMA = 0.01

# algorithm parameters
get_p_m = lambda l: 0.1 / l / N
get_pop_seed = lambda run_i: 1381*run_i + 5912826

# output parameters
DISTRIBUTIONS_TO_PLOT = 5
RUNS_TO_PLOT = 4 if N == 100 else 2
RUNS_TO_PLOT_K = 10000
RUNS_TO_PLOT_WITH_OP = 2 if N == 100 else 0
OUTPUT_FOLDER = 'out_test' if env == 'test' else 'out'

GEN_STATS_NAMES = [
    "f_avg", 'f_std', 'f_best', 'num_of_best', 'optimal_count',
    'growth_rate', 'difference', 'intensity', 'reproduction_rate',
    'loss_of_diversity', 'unique_X', 'Pr', 'Pr1', 'Pfet', 'Ptau'
]

FCONSTALL_GEN_STATS_NAMES = [
    'reproduction_rate', 'loss_of_diversity', 'unique_X'
]

RUN_STATS_NAMES = [
    'NI', 'F_found', 'F_avg',
    'NI_loose', 'optSaved_NI_loose', 'Num_loose',
    'RR_start', 'RR_fin', 'RR_min', 'NI_RR_min', 'RR_max', 'NI_RR_max', 'RR_avg',
    'Teta_start', 'Teta_fin', 'Teta_min', 'NI_Teta_min', 'Teta_max', 'NI_Teta_max', 'Teta_avg',
    'I_start', 'I_min', 'NI_I_min', 'I_max', 'NI_I_max', 'I_avg',
    's_start', 's_min', 'NI_s_min', 's_max', 'NI_s_max', 's_avg',
    'GR_start', 'GR_early', 'GR_late', 'NI_GR_late', 'GR_avg',
    'Pr_start', 'Pr_min', 'NI_Pr_min', 'Pr_max', 'NI_Pr_max', 'Pr_avg',
    'Fish_start', 'Fish_min', 'NI_Fish_min', 'Fish_max', 'NI_Fish_max', 'Fish_avg',
    'Kend_start', 'Kend_min', 'NI_Kend_min', 'Kend_max', 'NI_Kend_max', 'Kend_avg',
    'unique_X_start', 'unique_X_fin'
]
EXP_STATS_NAMES = [
    'Suc', 'N_Suc', 'Min_NI', 'Max_NI', 'Avg_NI', 'Sigma_NI',
    
    'Min_RR_min', 'NI_RR_min', 'Max_RR_max', 'NI_RR_max',
    'Avg_RR_min', 'Avg_RR_max', 'Avg_RR_avg',
    'Sigma_RR_min', 'Sigma_RR_max', 'Sigma_RR_avg',
    'Avg_RR_start', 'Avg_RR_fin',
    'Sigma_RR_start', 'Sigma_RR_fin',
    'Min_RR_start', 'Max_RR_start',

    'Min_Teta_min', 'NI_Teta_min', 'Max_Teta_max', 'NI_Teta_max',
    'Avg_Teta_min', 'Avg_Teta_max', 'Avg_Teta_avg',
    'Sigma_Teta_min', 'Sigma_Teta_max', 'Sigma_Teta_avg',
    'Avg_Teta_start', 'Avg_Teta_fin',
    'Sigma_Teta_start', 'Sigma_Teta_fin',
    'Min_Teta_start', 'Max_Teta_start',
    
    'Min_I_min', 'NI_I_min', 'Max_I_max', 'NI_I_max',
    'Avg_I_min', 'Avg_I_max', 'Avg_I_avg',
    'Sigma_I_min', 'Sigma_I_max', 'Sigma_I_avg',
    'Min_I_start', 'Max_I_start', 'Avg_I_start', 'Sigma_I_start',
    
    'Min_s_min', 'NI_s_min', 'Max_s_max', 'NI_s_max',
    'Avg_s_min', 'Avg_s_max', 'Avg_s_avg',
    'Min_s_start', 'Max_s_start', 'Avg_s_start', 'Sigma_s_start',
    
    'Min_GR_early', 'Max_GR_early', 'Avg_GR_early',
    'Min_GR_late', 'Max_GR_late', 'Avg_GR_late',
    'Min_GR_avg', 'Max_GR_avg', 'Avg_GR_avg',
    'Sigma_GR_early', 'Sigma_GR_late', 'Sigma_GR_avg',
    'Min_GR_start', 'Max_GR_start', 'Avg_GR_start', 'Sigma_GR_start',

    'Avg_NI_loose', 'Sigma_NI_loose', 'Avg_optSaved_NI_loose', 'Sigma_optSaved_NI_loose',
    'Avg_Num_loose', 'Sigma_Num_loose',
    
    'nonSuc_N', 'nonSuc', 'nonMin_NI', 'nonMax_NI',
    'nonAvg_NI', 'nonSigma_NI', 'nonAvg_F_found', 'nonSigma_F_found',
    'nonMax_F_found', 'nonAvg_NI_loose', 'nonSigma_NI_loose',
    'nonAvg_optSaved_NI_loose', 'nonSigma_optSaved_NI_loose',
    'nonAvg_Num_loose', 'nonSigma_Num_loose',

    'Min_Pr_min', 'NI_Pr_min', 'Max_Pr_max', 'NI_Pr_max',
    'Avg_Pr_min', 'Avg_Pr_avg', 'Sigma_Pr_min', 'Sigma_Pr_avg',
    'Min_Pr_start', 'Max_Pr_start', 'Avg_Pr_start', 'Sigma_Pr_start',

    'Min_Fish_min', 'NI_Fish_min', 'Max_Fish_max', 'NI_Fish_max',
    'Avg_Fish_min', 'Avg_Fish_avg', 'Sigma_Fish_max', 'Sigma_Fish_min',
    'Sigma_Fish_avg', 'Min_Fish_start', 'Max_Fish_start',
    'Avg_Fish_start', 'Sigma_Fish_start',
    
    'Min_Kend_min', 'NI_Kend_min', 'Max_Kend_max', 'NI_Kend_max',
    'Avg_Kend_min', 'Avg_Kend_avg', 'Sigma_Kend_max', 'Sigma_Kend_min',
    'Sigma_Kend_avg', 'Min_Kend_start', 'Max_Kend_start', 'Avg_Kend_start',
    'Sigma_Kend_start',

    'Avg_unique_X_start', 'Avg_unique_X_fin', 'Sigma_unique_X_start',
    'Sigma_unique_X_fin', 'Min_unique_X_start', 'Max_unique_X_start',
    'Min_unique_X_fin', 'Max_unique_X_fin'
]
FCONSTALL_RUN_STATS_NAMES = [
    'NI',
    'RR_start', 'RR_fin', 'RR_min', 'NI_RR_min', 'RR_max', 'NI_RR_max', 'RR_avg',
    'Teta_start', 'Teta_fin', 'Teta_min', 'NI_Teta_min', 'Teta_max', 'NI_Teta_max', 'Teta_avg',
    'unique_X_start', 'unique_X_fin'
]
FCONSTALL_EXP_STATS_NAMES = [
    'Suc', 'N_Suc', 'Min_NI', 'Max_NI', 'Avg_NI', 'Sigma_NI',
    
     'Min_RR_min', 'NI_RR_min', 'Max_RR_max', 'NI_RR_max',
    'Avg_RR_min', 'Avg_RR_max', 'Avg_RR_avg',
    'Sigma_RR_min', 'Sigma_RR_max', 'Sigma_RR_avg',
    'Avg_RR_start', 'Avg_RR_fin',
    'Sigma_RR_start', 'Sigma_RR_fin',
    'Min_RR_start', 'Max_RR_start',

    'Min_Teta_min', 'NI_Teta_min', 'Max_Teta_max', 'NI_Teta_max',
    'Avg_Teta_min', 'Avg_Teta_max', 'Avg_Teta_avg',
    'Sigma_Teta_min', 'Sigma_Teta_max', 'Sigma_Teta_avg',
    'Avg_Teta_start', 'Avg_Teta_fin',
    'Sigma_Teta_start', 'Sigma_Teta_fin',
    'Min_Teta_start', 'Max_Teta_start',

    'Avg_unique_X_start', 'Avg_unique_X_fin', 'Sigma_unique_X_start',
    'Sigma_unique_X_fin', 'Min_unique_X_start', 'Max_unique_X_start',
    'Min_unique_X_fin', 'Max_unique_X_fin'
]
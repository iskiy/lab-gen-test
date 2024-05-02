from config import NR
from stats.run_stats import RunStats
import numpy as np


class ExperimentStats:
    def __init__(self, experiment_params: tuple[str]):
        self.params = experiment_params
        self.runs = np.empty(NR, dtype=object)

        self.Suc = 0
        self.N_Suc = 0
        self.Min_NI = None
        self.Max_NI = None
        self.Avg_NI = None
        self.Sigma_NI = None
        # self.Sigma_NI_formula = None

        # Reproduction Rate
        self.Min_RR_min = None
        self.NI_RR_min = None
        self.Max_RR_max = None
        self.NI_RR_max = None
        self.Avg_RR_min = None
        self.Avg_RR_max = None
        self.Avg_RR_avg = None
        self.Sigma_RR_min = None
        self.Sigma_RR_max = None
        self.Sigma_RR_avg = None
        self.Avg_RR_start = None
        self.Avg_RR_fin = None
        self.Sigma_RR_start = None
        self.Sigma_RR_fin = None
        self.Min_RR_start = None
        self.Max_RR_start = None

        # Loss of Diversity
        self.Min_Teta_min = None
        self.NI_Teta_min = None
        self.Max_Teta_max = None
        self.NI_Teta_max = None
        self.Avg_Teta_min = None
        self.Avg_Teta_max = None
        self.Avg_Teta_avg = None
        self.Sigma_Teta_min = None
        self.Sigma_Teta_max = None
        self.Sigma_Teta_avg = None
        self.Avg_Teta_start = None
        self.Avg_Teta_fin = None
        self.Sigma_Teta_start = None
        self.Sigma_Teta_fin = None
        self.Min_Teta_start = None
        self.Max_Teta_start = None

        # Selection Intensity
        self.Min_I_min = None
        self.NI_I_min = None
        self.Max_I_max = None
        self.NI_I_max = None
        self.Avg_I_min = None
        self.Avg_I_max = None
        self.Avg_I_avg = None
        self.Sigma_I_min = None
        self.Sigma_I_max = None
        self.Sigma_I_avg = None
        self.Min_I_start = None
        self.Max_I_start = None
        self.Avg_I_start = None
        self.Sigma_I_start = None

        # Selection Difference
        self.Min_s_min = None
        self.NI_s_min = None
        self.Max_s_max = None
        self.NI_s_max = None
        self.Avg_s_min = None
        self.Avg_s_max = None
        self.Avg_s_avg = None
        self.Min_s_start = None
        self.Max_s_start = None
        self.Avg_s_start = None
        self.Sigma_s_start = None

        # Growth Rate
        self.Min_GR_early = None
        self.Max_GR_early = None
        self.Avg_GR_early = None
        self.Sigma_GR_early = None
        self.Min_GR_late = None
        self.Max_GR_late = None
        self.Avg_GR_late = None
        self.Sigma_GR_late = None
        self.Min_GR_avg = None
        self.Max_GR_avg = None
        self.Avg_GR_avg = None
        self.Sigma_GR_avg = None
        self.Min_GR_start = None
        self.Max_GR_start = None
        self.Avg_GR_start = None
        self.Sigma_GR_start = None

        # Optimal Loose
        self.Avg_NI_loose = None
        self.Sigma_NI_loose = None
        self.Avg_optSaved_NI_loose = None
        self.Sigma_optSaved_NI_loose = None
        self.Avg_Num_loose = None
        self.Sigma_Num_loose = None

        # Non suc
        self.nonSuc_N = None
        self.nonSuc = None
        self.nonMin_NI = None
        self.nonMax_NI = None
        self.nonAvg_NI = None
        self.nonSigma_NI = None
        self.nonAvg_F_found = None
        self.nonSigma_F_found = None
        self.nonMax_F_found = None
        self.nonAvg_NI_loose = None
        self.nonSigma_NI_loose = None
        self.nonAvg_optSaved_NI_loose = None
        self.nonSigma_optSaved_NI_loose = None
        self.nonAvg_Num_loose = None
        self.nonSigma_Num_loose = None

        # Pressure
        self.Min_Pr_min = None
        self.NI_Pr_min = None
        self.Max_Pr_max = None
        self.NI_Pr_max = None
        self.Avg_Pr_min = None
        self.Avg_Pr_avg = None
        self.Sigma_Pr_min = None
        self.Sigma_Pr_avg = None
        self.Min_Pr_start = None
        self.Max_Pr_start = None
        self.Avg_Pr_start = None
        self.Sigma_Pr_start = None

        # Fisher exact test
        self.Min_Fish_min = None
        self.NI_Fish_min = None
        self.Max_Fish_max = None
        self.NI_Fish_max = None
        self.Avg_Fish_min = None
        self.Avg_Fish_avg = None
        self.Sigma_Fish_max = None
        self.Sigma_Fish_min = None
        self.Sigma_Fish_avg = None
        self.Min_Fish_start = None
        self.Max_Fish_start = None
        self.Avg_Fish_start = None
        self.Sigma_Fish_start = None

        # Kendall tau
        self.Min_Kend_min = None
        self.NI_Kend_min = None
        self.Max_Kend_max = None
        self.NI_Kend_max = None
        self.Avg_Kend_min = None
        self.Avg_Kend_avg = None
        self.Sigma_Kend_max = None
        self.Sigma_Kend_min = None
        self.Sigma_Kend_avg = None
        self.Min_Kend_start = None
        self.Max_Kend_start = None
        self.Avg_Kend_start = None
        self.Sigma_Kend_start = None

        # Unique
        self.Avg_unique_X_start = None
        self.Avg_unique_X_fin = None
        self.Sigma_unique_X_start = None
        self.Sigma_unique_X_fin = None
        self.Min_unique_X_start = None
        self.Max_unique_X_start = None
        self.Min_unique_X_fin = None
        self.Max_unique_X_fin = None

    def add_run(self, run: RunStats, run_i):
        self.runs[run_i] = run

    def calculate(self):
        successful_runs = [run for run in self.runs if run.is_successful]
        self.N_Suc = len(successful_runs)
        self.Suc = self.N_Suc / NR

        self.__calculate_convergence_stats(successful_runs)
        self.__calculate_rr_stats(successful_runs)
        self.__calculate_teta_stats(successful_runs)
        self.__calculate_unique_stats(successful_runs)

        if self.params[0] != 'FconstALL':
            self.__calculate_optimal_loose_stats(successful_runs)
            self.__calculate_s_stats(successful_runs)
            self.__calculate_i_stats(successful_runs)
            self.__calculate_pr_stats(successful_runs)
            self.__calculate_fish_stats(successful_runs)
            self.__calculate_kendall_stats(successful_runs)
            self.__calculate_gr_stats(successful_runs)

        non_successful_runs = [run for run in self.runs if not run.is_successful and run.is_converged]
        self.nonSuc_N = len(non_successful_runs)
        self.nonSuc = self.nonSuc_N / NR

        if self.params[0] != 'FconstALL':
            self.__calculate_non_optimal_loose_stats(non_successful_runs)
            self.__calculate_non_convergence_stats(non_successful_runs)
            self.__calculate_non_found_stats(non_successful_runs)

    def get_selection_noise(self):
        return self.Suc, self.Min_NI, self.Max_NI, self.Avg_NI, self.Sigma_NI

    def get_convergence(self):
        return self.Suc, self.Min_NI, self.Max_NI, self.Avg_NI, self.Sigma_NI, \
            self.Avg_NI_loose, self.Sigma_NI_loose, \
            self.Avg_optSaved_NI_loose, self.Sigma_optSaved_NI_loose,\
            self.Avg_Num_loose, self.Sigma_Num_loose, \
            self.nonSuc, self.nonMin_NI, self.nonMax_NI, self.nonAvg_NI, self.nonSigma_NI, \
            self.nonAvg_F_found, self.nonSigma_F_found, self.nonMax_F_found, \
            self.nonAvg_NI_loose, self.nonSigma_NI_loose, \
            self.nonAvg_optSaved_NI_loose, self.nonSigma_optSaved_NI_loose, \
            self.nonAvg_Num_loose, self.nonSigma_Num_loose

    def __calculate_non_found_stats(self, runs: list[RunStats]):
        F_founds = [run.F_found for run in runs]

        if F_founds:
            self.nonMax_F_found = max(F_founds)
            self.nonAvg_F_found = np.mean(F_founds)
            self.nonSigma_F_found = np.std(F_founds)

    def __calculate_optimal_loose_stats(self, runs: list[RunStats]):
        NI_looses = [run.NI_loose for run in runs]
        optSaved_NI_looses = [run.optSaved_NI_loose for run in runs]
        Num_looses = [run.Num_loose for run in runs]

        if NI_looses:
            self.Avg_NI_loose = np.mean(NI_looses)
            self.Sigma_NI_loose = np.std(NI_looses)

        if optSaved_NI_looses:
            self.Avg_optSaved_NI_loose = np.mean(optSaved_NI_looses)
            self.Sigma_optSaved_NI_loose = np.std(optSaved_NI_looses)

        if Num_looses:
            self.Avg_Num_loose = np.mean(Num_looses)
            self.Sigma_Num_loose = np.std(Num_looses)

    def __calculate_non_optimal_loose_stats(self, runs: list[RunStats]):
        NI_looses = [run.NI_loose for run in runs]
        optSaved_NI_looses = [run.optSaved_NI_loose for run in runs]
        Num_looses = [run.Num_loose for run in runs]

        if NI_looses:
            self.nonAvg_NI_loose = np.mean(NI_looses)
            self.nonSigma_NI_loose = np.std(NI_looses)

        if optSaved_NI_looses:
            self.nonAvg_optSaved_NI_loose = np.mean(optSaved_NI_looses)
            self.nonSigma_optSaved_NI_loose = np.std(optSaved_NI_looses)

        if Num_looses:
            self.nonAvg_Num_loose = np.mean(Num_looses)
            self.nonSigma_Num_loose = np.std(Num_looses)

    def __calculate_convergence_stats(self, runs: list[RunStats]):
        NIs = [run.NI for run in runs]
        if NIs:
            self.Min_NI = min(NIs)
            self.Max_NI = max(NIs)
            self.Avg_NI = np.mean(NIs)
            self.Sigma_NI = np.std(NIs)
            # suma = 0
            # n = 0
            # for run in runs:
            #     if run.is_successful:
            #         suma += (run.NI - self.Avg_NI) ** 2
            #         n += 1
            # self.Sigma_NI_formula = 1./(n - 1) * suma

    def __calculate_non_convergence_stats(self, runs: list[RunStats]):
        NIs = [run.NI for run in runs]
        if NIs:
            self.nonMin_NI = min(NIs)
            self.nonMax_NI = max(NIs)
            self.nonAvg_NI = np.mean(NIs)
            self.nonSigma_NI = np.std(NIs)

    def __calculate_rr_stats(self, runs: list[RunStats]):
        RR_min_list = [run.RR_min for run in runs]
        if RR_min_list:
            run_i_RR_min = np.argmin(RR_min_list)
            self.NI_RR_min = runs[run_i_RR_min].NI_RR_min
            self.Min_RR_min = RR_min_list[run_i_RR_min]
            self.Avg_RR_min = np.mean(RR_min_list)
            self.Sigma_RR_min = np.std(RR_min_list)
        RR_max_list = [run.RR_max for run in runs]
        if RR_max_list:
            run_i_RR_max = np.argmax(RR_max_list)
            self.NI_RR_max = runs[run_i_RR_max].NI_RR_max
            self.Max_RR_max = RR_max_list[run_i_RR_max]
            self.Avg_RR_max = np.mean(RR_max_list)
            self.Sigma_RR_max = np.std(RR_max_list)
        RR_avg_list = [run.RR_avg for run in runs]
        if RR_avg_list:
            self.Avg_RR_avg = np.mean(RR_avg_list)
            self.Sigma_RR_avg = np.std(RR_avg_list)
        RR_start_list = [run.RR_start for run in runs]
        if RR_start_list:
            self.Avg_RR_start = np.mean(RR_start_list)
            self.Sigma_RR_start = np.std(RR_start_list)
            self.Min_RR_start = min(RR_start_list)
            self.Max_RR_start = max(RR_start_list)
        RR_fin_list = [run.RR_fin for run in runs]
        if RR_fin_list:
            self.Avg_RR_fin = np.mean(RR_fin_list)
            self.Sigma_RR_fin = np.std(RR_fin_list)

    def __calculate_teta_stats(self, runs: list[RunStats]):
        Teta_min_list = [run.Teta_min for run in runs]
        if Teta_min_list:
            run_i_Teta_min = np.argmin(Teta_min_list)
            self.NI_Teta_min = runs[run_i_Teta_min].NI_Teta_min
            self.Min_Teta_min = Teta_min_list[run_i_Teta_min]
            self.Avg_Teta_min = np.mean(Teta_min_list)
            self.Sigma_Teta_min = np.std(Teta_min_list)
        Teta_max_list = [run.Teta_max for run in runs]
        if Teta_max_list:
            run_i_Teta_max = np.argmax(Teta_max_list)
            self.NI_Teta_max = runs[run_i_Teta_max].NI_Teta_max
            self.Max_Teta_max = Teta_max_list[run_i_Teta_max]
            self.Avg_Teta_max = np.mean(Teta_max_list)
            self.Sigma_Teta_max = np.std(Teta_max_list)
        Teta_avg_list = [run.Teta_avg for run in runs]
        if Teta_avg_list:
            self.Avg_Teta_avg = np.mean(Teta_avg_list)
            self.Sigma_Teta_avg = np.std(Teta_avg_list)
        Teta_start_list = [run.Teta_start for run in runs]
        if Teta_start_list:
            self.Avg_Teta_start = np.mean(Teta_start_list)
            self.Sigma_Teta_start = np.std(Teta_start_list)
            self.Min_Teta_start = min(Teta_start_list)
            self.Max_Teta_start = max(Teta_start_list)
        Teta_fin_list = [run.Teta_fin for run in runs]
        if Teta_fin_list:
            self.Avg_Teta_fin = np.mean(Teta_fin_list)
            self.Sigma_Teta_fin = np.std(Teta_fin_list)

    def __calculate_s_stats(self, runs: list[RunStats]):
        s_min_list = [run.s_min for run in runs]
        if s_min_list:
            run_i_s_min = np.argmin(s_min_list)
            self.NI_s_min = runs[run_i_s_min].NI_s_min
            self.Min_s_min = s_min_list[run_i_s_min]
            self.Avg_s_min = np.mean(s_min_list)
        s_max_list = [run.s_max for run in runs]
        if s_max_list:
            run_i_s_max = np.argmax(s_max_list)
            self.NI_s_max = runs[run_i_s_max].NI_s_max
            self.Max_s_max = s_max_list[run_i_s_max]
            self.Avg_s_max = np.mean(s_max_list)
        s_avg_list = [run.s_avg for run in runs]
        if s_avg_list:
            self.Avg_s_avg = np.mean(s_avg_list)
        s_start_list = [run.s_start for run in runs]
        if s_start_list:
            self.Avg_s_start = np.mean(s_start_list)
            self.Sigma_s_start = np.std(s_start_list)
            self.Max_s_start = max(s_start_list)
            self.Min_s_start = min(s_start_list)

    def __calculate_i_stats(self, runs: list[RunStats]):
        I_min_list = [run.I_min for run in runs]
        if I_min_list:
            run_i_I_min = np.argmin(I_min_list)
            self.NI_I_min = runs[run_i_I_min].NI_I_min
            self.Min_I_min = I_min_list[run_i_I_min]
            self.Avg_I_min = np.mean(I_min_list)
            self.Sigma_I_min = np.std(I_min_list)
        I_max_list = [run.I_max for run in runs]
        if I_max_list:
            run_i_I_max = np.argmax(I_max_list)
            self.NI_I_max = runs[run_i_I_max].NI_I_max
            self.Max_I_max = I_max_list[run_i_I_max]
            self.Avg_I_max = np.mean(I_max_list)
            self.Sigma_I_max = np.std(I_max_list)
        I_avg_list = [run.I_avg for run in runs]
        if I_avg_list:
            self.Avg_I_avg = np.mean(I_avg_list)
            self.Sigma_I_avg = np.std(I_avg_list)
        I_start_list = [run.I_start for run in runs]
        if I_start_list:
            self.Min_I_start = min(I_start_list)
            self.Max_I_start = max(I_start_list)
            self.Avg_I_start = np.mean(I_start_list)
            self.Sigma_I_start = np.std(I_start_list)


    def __calculate_gr_stats(self, runs: list[RunStats]):
        gre_list = [run.GR_early for run in runs]
        grl_list = [run.GR_late for run in runs if run.GR_late is not None]
        gra_list = [run.GR_avg for run in runs]
        grs_list = [run.GR_start for run in runs]
        if gre_list:
            self.Avg_GR_early = np.mean(gre_list)
            self.Min_GR_early = min(gre_list)
            self.Max_GR_early = max(gre_list)
            self.Sigma_GR_early = np.std(gre_list)
        if grl_list:
            self.Avg_GR_late = np.mean(grl_list)
            self.Min_GR_late = min(grl_list)
            self.Max_GR_late = max(grl_list)
            self.Sigma_GR_late = np.std(grl_list)
        if gra_list:
            self.Avg_GR_avg = np.mean(gra_list)
            self.Min_GR_avg = min(gra_list)
            self.Max_GR_avg = max(gra_list)
            self.Sigma_GR_avg = np.std(gra_list)
        if grs_list:
            self.Avg_GR_start = np.mean(grs_list)
            self.Min_GR_start = min(grs_list)
            self.Max_GR_start = max(grs_list)
            self.Sigma_GR_start = np.std(grs_list)

    def __calculate_pr_stats(self, runs: list[RunStats]):
        Pr_min_list = [run.Pr_min for run in runs]
        if Pr_min_list:
            run_i_Pr_min = np.argmin(Pr_min_list)
            self.NI_Pr_min = runs[run_i_Pr_min].NI_Pr_min
            self.Min_Pr_min = Pr_min_list[run_i_Pr_min]
            self.Avg_Pr_min = np.mean(Pr_min_list)
            self.Sigma_Pr_min = np.std(Pr_min_list)
        Pr_max_list = [run.Pr_max for run in runs]
        if Pr_max_list:
            run_i_Pr_max = np.argmax(Pr_max_list)
            self.NI_PR_max = runs[run_i_Pr_max].NI_Pr_max
            self.Max_Pr_max = Pr_max_list[run_i_Pr_max]
            self.Avg_Pr_max = np.mean(Pr_max_list)
            self.Sigma_Pr_max = np.std(Pr_max_list)
        Pr_avg_list = [run.Pr_avg for run in runs]
        if Pr_avg_list:
            self.Avg_Pr_avg = np.mean(Pr_avg_list)
            self.Sigma_Pr_avg = np.std(Pr_avg_list)
        Pr_start_list = [run.Pr_start for run in runs]
        if Pr_start_list:
            self.Min_Pr_start = min(Pr_start_list)
            self.Max_Pr_start = max(Pr_start_list)
            self.Avg_Pr_start = np.mean(Pr_start_list)
            self.Sigma_Pr_start = np.std(Pr_start_list)

    def __calculate_fish_stats(self, runs: list[RunStats]):
        Fish_min_list = [run.Fish_min for run in runs]
        if Fish_min_list:
            run_i_Fish_min = np.argmin(Fish_min_list)
            self.NI_Fish_min = runs[run_i_Fish_min].NI_Fish_min
            self.Min_Fish_min = Fish_min_list[run_i_Fish_min]
            self.Avg_Fish_min = np.mean(Fish_min_list)
            self.Sigma_Fish_min = np.std(Fish_min_list)
        Fish_max_list = [run.Fish_max for run in runs]
        if Fish_max_list:
            run_i_Fish_max = np.argmax(Fish_max_list)
            self.NI_Fish_max = runs[run_i_Fish_max].NI_Fish_max
            self.Max_Fish_max = Fish_max_list[run_i_Fish_max]
            self.Avg_Fish_max = np.mean(Fish_max_list)
            self.Sigma_Fish_max = np.std(Fish_max_list)
        Fish_avg_list = [run.Fish_avg for run in runs]
        if Fish_avg_list:
            self.Avg_Fish_avg = np.mean(Fish_avg_list)
            self.Sigma_Fish_avg = np.std(Fish_avg_list)
        Fish_start_list = [run.Fish_start for run in runs]
        if Fish_start_list:
            self.Min_Fish_start = min(Fish_start_list)
            self.Max_Fish_start = max(Fish_start_list)
            self.Avg_Fish_start = np.mean(Fish_start_list)
            self.Sigma_Fish_start = np.std(Fish_start_list)

    def __calculate_kendall_stats(self, runs: list[RunStats]):
        Kend_min_list = [run.Kend_min for run in runs]
        if Kend_min_list:
            run_i_Kend_min = np.argmin(Kend_min_list)
            self.NI_Kend_min = runs[run_i_Kend_min].NI_Kend_min
            self.Min_Kend_min = Kend_min_list[run_i_Kend_min]
            self.Avg_Kend_min = np.mean(Kend_min_list)
            self.Sigma_Kend_min = np.std(Kend_min_list)
        Kend_max_list = [run.Kend_max for run in runs]
        if Kend_max_list:
            run_i_Kend_max = np.argmax(Kend_max_list)
            self.NI_Kend_max = runs[run_i_Kend_max].NI_Kend_max
            self.Max_Kend_max = Kend_max_list[run_i_Kend_max]
            self.Avg_Kend_max = np.mean(Kend_max_list)
            self.Sigma_Kend_max = np.std(Kend_max_list)
        Kend_avg_list = [run.Kend_avg for run in runs]
        if Kend_avg_list:
            self.Avg_Kend_avg = np.mean(Kend_avg_list)
            self.Sigma_Kend_avg = np.std(Kend_avg_list)
        Kend_start_list = [run.Kend_start for run in runs]
        if Kend_start_list:
            self.Min_Kend_start = min(Kend_start_list)
            self.Max_Kend_start = max(Kend_start_list)
            self.Avg_Kend_start = np.mean(Kend_start_list)
            self.Sigma_Kend_start = np.std(Kend_start_list)

    def __calculate_unique_stats(self, runs: list[RunStats]):
        unique_start_list = [run.unique_X_start for run in runs]
        unique_fin_list = [run.unique_X_fin for run in runs]
        if unique_start_list:
            self.Avg_unique_X_start = np.mean(unique_start_list)
            self.Min_unique_X_start = min(unique_start_list)
            self.Max_unique_X_start = max(unique_start_list)
            self.Sigma_unique_X_start = np.std(unique_start_list)
        if unique_fin_list:
            self.Avg_unique_X_fin = np.mean(unique_fin_list)
            self.Min_unique_X_fin = min(unique_fin_list)
            self.Max_unique_X_fin = max(unique_fin_list)
            self.Sigma_unique_X_fin = np.std(unique_fin_list)

    def __str__(self):
        return ("Suc: " + str(self.Suc) + "%" +
                "\nMin: " + str(self.Min_NI) + "\nMax: " + str(self.Max_NI) + "\nAvg: " + str(self.Avg_NI))

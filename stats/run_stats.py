from config import N
from stats.generation_stats import GenerationStats

class RunStats:
    def __init__(self, param_names: tuple[str]):
        self.param_names = param_names

        self.NI = None
        self.F_found = None
        self.F_avg = None
        self.is_successful = False
        self.is_converged = False

        self.NI_loose = 0
        self.optSaved_NI_loose = None
        self.Num_loose = 0

        # Reproduction Rate
        self.RR_start = None
        self.RR_fin = None
        self.RR_min = None
        self.NI_RR_min = None
        self.RR_max = None
        self.NI_RR_max = None
        self.RR_avg = None

        # Loss of Diversity
        self.Teta_start = None
        self.Teta_fin = None
        self.Teta_min = None
        self.NI_Teta_min = None
        self.Teta_max = None
        self.NI_Teta_max = None
        self.Teta_avg = None

        # Selection Intensity
        self.I_start = None
        self.I_min = None
        self.NI_I_min = None
        self.I_max = None
        self.NI_I_max = None
        self.I_avg = None

        # Selection Difference
        self.s_min = None
        self.NI_s_min = None
        self.s_max = None
        self.NI_s_max = None
        self.s_avg = None
        self.s_start = None

        # Growth Rate
        self.GR_start = None
        self.GR_early = None
        self.GR_late = None
        self.NI_GR_late = None
        self.GR_avg = None

        # Pressure
        self.Pr_start = None
        self.Pr_min = None
        self.NI_Pr_min = None
        self.Pr_max = None
        self.NI_Pr_max = None
        self.Pr_avg = None

        # Fisher exact test
        self.Fish_start = None
        self.Fish_min = None
        self.NI_Fish_min = None
        self.Fish_max = None
        self.NI_Fish_max = None
        self.Fish_avg = None

        # Kendall tau
        self.Kend_start = None
        self.Kend_min = None
        self.NI_Kend_min = None
        self.Kend_max = None
        self.NI_Kend_max = None
        self.Kend_avg = None

        self.unique_X_start = None
        self.unique_X_fin = None

        self.optimal_count = None

    def update_stats_for_generation(self, gen_stats: GenerationStats, gen_i):
        # Reproduction Rate
        if self.RR_start is None:
            self.RR_start = gen_stats.reproduction_rate
        self.RR_fin = gen_stats.reproduction_rate
        if self.RR_min is None or gen_stats.reproduction_rate < self.RR_min:
            self.RR_min = gen_stats.reproduction_rate
            self.NI_RR_min = gen_i
        if self.RR_max is None or gen_stats.reproduction_rate > self.RR_max:
            self.RR_max = gen_stats.reproduction_rate
            self.NI_RR_max = gen_i
        if self.RR_avg is None:
            self.RR_avg = gen_stats.reproduction_rate
        else:
            self.RR_avg = (self.RR_avg * (gen_i - 1) + gen_stats.reproduction_rate) / gen_i

        # Loss of Diversity
        if self.Teta_start is None:
            self.Teta_start = gen_stats.loss_of_diversity
        self.Teta_fin = gen_stats.loss_of_diversity
        if self.Teta_min is None or gen_stats.loss_of_diversity < self.Teta_min:
            self.Teta_min = gen_stats.loss_of_diversity
            self.NI_Teta_min = gen_i
        if self.Teta_max is None or gen_stats.loss_of_diversity > self.Teta_max:
            self.Teta_max = gen_stats.loss_of_diversity
            self.NI_Teta_max = gen_i
        if self.Teta_avg is None:
            self.Teta_avg = gen_stats.loss_of_diversity
        else:
            self.Teta_avg = (self.Teta_avg * (gen_i - 1) + gen_stats.loss_of_diversity) / gen_i

        if self.unique_X_start is None:
            self.unique_X_start = gen_stats.unique_X
        self.unique_X_fin = gen_stats.unique_X

        if self.param_names[0] != 'FconstALL':
            # Selection Intensity
            if self.I_start is None:
                self.I_start = gen_stats.intensity
            if self.I_min is None or gen_stats.intensity < self.I_min:
                self.I_min = gen_stats.intensity
                self.NI_I_min = gen_i
            if self.I_max is None or gen_stats.intensity > self.I_max:
                self.I_max = gen_stats.intensity
                self.NI_I_max = gen_i
            if self.I_avg is None:
                self.I_avg = gen_stats.intensity
            else:
                self.I_avg = (self.I_avg * (gen_i - 1) + gen_stats.intensity) / gen_i

            # Selection Difference
            if self.s_start is None:
                self.s_start = gen_stats.difference
            if self.s_min is None or gen_stats.difference < self.s_min:
                self.s_min = gen_stats.difference
                self.NI_s_min = gen_i
            if self.s_max is None or gen_stats.difference > self.s_max:
                self.s_max = gen_stats.difference
                self.NI_s_max = gen_i
            if self.s_avg is None:
                self.s_avg = gen_stats.difference
            else:
                self.s_avg = (self.s_avg * (gen_i - 1) + gen_stats.difference) / gen_i

            # Growth Rate
            if gen_i == 2:
                self.GR_early = gen_stats.growth_rate
            if self.GR_start is None:
                self.GR_start = gen_stats.growth_rate
            if self.GR_late is None and gen_stats.num_of_best >= N / 2:
                self.GR_late = gen_stats.growth_rate
                self.NI_GR_late = gen_i
            if self.GR_avg is None:
                self.GR_avg = gen_stats.growth_rate
            else:
                self.GR_avg = (self.GR_avg * (gen_i - 1) + gen_stats.growth_rate) / gen_i

            # Pressure
            # TODO check
            if self.Pr_start is None:
                self.Pr_start = gen_stats.Pr
            if self.Pr_min is None or gen_stats.Pr < self.Pr_min:
                self.Pr_min = gen_stats.Pr
                self.NI_Pr_min = gen_i
            if self.Pr_max is None or gen_stats.Pr > self.Pr_max:
                self.Pr_max = gen_stats.Pr
                self.NI_Pr_max = gen_i
            if self.Pr_avg is None:
                self.Pr_avg = gen_stats.Pr
            else:
                self.Pr_avg = (self.Pr_avg * (gen_i - 1) + gen_stats.Pr) / gen_i

            # Fisher exact test
            # TODO check
            if self.Fish_start is None:
                self.Fish_start = gen_stats.Pfet
            if self.Fish_min is None or gen_stats.Pfet < self.Fish_min:
                self.Fish_min = gen_stats.Pfet
                self.NI_Fish_min = gen_i
            if self.Fish_max is None or gen_stats.Pfet > self.Fish_max:
                self.Fish_max = gen_stats.Pfet
                self.NI_Fish_max = gen_i
            if self.Fish_avg is None:
                self.Fish_avg = gen_stats.Pfet
            else:
                self.Fish_avg = (self.Fish_avg * (gen_i - 1) + gen_stats.Pfet) / gen_i

            # Kendall tau
            # TODO check
            if self.Kend_start is None:
                self.Kend_start = gen_stats.Ptau
            if self.Kend_min is None or gen_stats.Ptau < self.Kend_min:
                self.Kend_min = gen_stats.Ptau
                self.NI_Kend_min = gen_i
            if self.Kend_max is None or gen_stats.Ptau > self.Kend_max:
                self.Kend_max = gen_stats.Ptau
                self.NI_Kend_max = gen_i
            if self.Kend_avg is None:
                self.Kend_avg = gen_stats.Ptau
            else:
                self.Kend_avg = (self.Kend_avg * (gen_i - 1) + gen_stats.Ptau) / gen_i

            # Todo check
            if gen_i > 0:
                if self.optimal_count > gen_stats.optimal_count and gen_stats.optimal_count == 0:
                    self.NI_loose = gen_i
                    self.optSaved_NI_loose = gen_stats.optimal_count
                    self.Num_loose += 1
            else:
                self.NI_loose = 0
                self.optSaved_NI_loose = gen_stats.optimal_count
                self.Num_loose = 0
            self.optimal_count = gen_stats.optimal_count

    def update_final_stats(self, gen_stats: GenerationStats, gen_i):
        if self.unique_X_start is None:
            self.unique_X_start = gen_stats.unique_X
        self.unique_X_fin = gen_stats.unique_X

        if self.param_names[0] != 'FconstALL':
            self.F_found = gen_stats.f_best
            self.F_avg = gen_stats.f_avg

            if gen_i == 2:
                self.GR_early = gen_stats.growth_rate
            if self.GR_late is None and gen_stats.num_of_best >= N / 2:
                self.GR_late = gen_stats.growth_rate
                self.NI_GR_late = gen_i
            if self.GR_avg is None:
                self.GR_avg = gen_stats.growth_rate
            else:
                self.GR_avg = (self.GR_avg * (gen_i - 1) + gen_stats.growth_rate) / gen_i

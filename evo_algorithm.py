from config import *
from model.fitness_functions import *
from selection.selection_method import SelectionMethod
from model.gen_operators import GeneticOperator
from stats.run_stats import RunStats
from stats.generation_stats import GenerationStats
from output import plotting
from output import excel
from model.population import *

class EvoAlgorithm:
    def __init__(self,
                 initial_population: Population,
                 selection_method: SelectionMethod,
                 genetic_operator: GeneticOperator,
                 param_names: tuple[str], plotter):
        self.population: Population = initial_population
        self.selection_method = selection_method
        self.genetic_operator = genetic_operator
        self.param_names = param_names

        self.gen_i = 0
        self.run_stats = RunStats(self.param_names)
        self.prev_gen_stats = None
        self.gen_stats_list = None
        self.has_converged = False
        self.plotter = plotter
        self.plotted_99 = False
        self.plotted_95 = False
        self.plotted_90 = False
        self.plotted_80 = False
        self.plotted_70 = False

    def run(self, run_i):
        # is_with_operator = has_operator(self.param_names[2])
        has_operator = check_has_operator(self.param_names[2])
        if not has_operator and run_i < RUNS_TO_PLOT or has_operator and run_i < RUNS_TO_PLOT_WITH_OP:
            self.gen_stats_list = []

        f_avgs = []
        while not self.has_converged and self.gen_i < G:
            gen_stats = self.__calculate_stats_and_evolve(run_i)

            f_avgs.append(gen_stats.f_avg)
            if len(f_avgs) > N_LAST_GENS:
                f_avgs.pop(0)

            self.has_converged = self.population.has_converged(f_avgs, self.param_names)
            self.prev_gen_stats = gen_stats
            self.gen_i += 1

        gen_stats = self.__calculate_final_stats(run_i)
        self.run_stats.NI = self.gen_i
        self.run_stats.is_successful = self.__check_success(gen_stats)
        self.run_stats.is_converged = self.has_converged

        if not has_operator and run_i < RUNS_TO_PLOT or has_operator and run_i < RUNS_TO_PLOT_WITH_OP:
            # self.plotter.plot_generation_stats(self.population, self.param_names, run_i, self.gen_i, "start2")
            self.plotter.plot_run_stats(self.gen_stats_list, self.param_names, run_i)
            excel.write_run_stats(self.gen_stats_list, self.param_names, run_i, "stats")

        return self.run_stats

    def __calculate_stats_and_evolve(self, run_i):
        self.plotted = False
        has_operator = check_has_operator(self.param_names[2])
        can_plot = not has_operator and run_i < RUNS_TO_PLOT or has_operator and run_i < RUNS_TO_PLOT_WITH_OP
        if can_plot and self.gen_i < DISTRIBUTIONS_TO_PLOT:
            self.plotter.plot_generation_stats(self.population, self.param_names, run_i, self.gen_i, "start")
            self.plotted = True
        elif can_plot and run_i != 0 and run_i % RUNS_TO_PLOT_K == 0:
            self.plotter.plot_generation_stats(self.population, self.param_names, run_i, self.gen_i, "K10000")
            self.plotted = True

        gen_stats = GenerationStats(self.population, self.param_names)
        if can_plot:
            self.gen_stats_list.append(gen_stats)

        gen_stats.calculate_stats_before_selection(self.prev_gen_stats)
        # TODO check

        if can_plot:
            if not self.plotted_99 and not self.plotted and self.population.is_homogenous_99():
                self.plotter.plot_generation_stats(self.population, self.param_names, run_i, self.gen_i, "homogenous99")
                excel.write_population(self.population, self.param_names, run_i, self.gen_i, "homogenous99")
                self.plotted_99 = True
                self.plotted = True
            elif not self.plotted_95 and not self.plotted and self.population.is_homogenous_percent(0.95):
                self.plotter.plot_generation_stats(self.population, self.param_names, run_i, self.gen_i, "homogenous95")
                excel.write_population(self.population, self.param_names, run_i, self.gen_i, "homogenous95")
                self.plotted_95 = True
                self.plotted = True
            elif not self.plotted_90 and not self.plotted and self.population.is_homogenous_percent(0.90):
                self.plotter.plot_generation_stats(self.population, self.param_names, run_i, self.gen_i, "homogenous90")
                excel.write_population(self.population, self.param_names, run_i, self.gen_i, "homogenous90")
                self.plotted = True
                self.plotted_90 = True
            elif not self.plotted_80 and not self.plotted and self.population.is_homogenous_percent(0.80):
                self.plotter.plot_generation_stats(self.population, self.param_names, run_i, self.gen_i, "homogenous80")
                # excel.write_run_stats(self.gen_stats_list, self.param_names, run_i, "homogenous80")
                self.plotted = True
                self.plotted_80 = True
            elif not self.plotted_70 and not self.plotted and self.population.is_homogenous_percent(0.70):
                self.plotter.plot_generation_stats(self.population, self.param_names, run_i, self.gen_i, "homogenous70")
                # excel.write_run_stats(self.gen_stats_list, self.param_names, run_i, "homogenous70")
                self.plotted = True
                self.plotted_70 = True
            # elif not self.plotted and self.population.is_homogenous_percent(0.50):
            #     self.plotter.plot_generation_stats(self.population, self.param_names, run_i, self.gen_i, "homogenous50")
            #     # excel.write_run_stats(self.gen_stats_list, self.param_names, run_i, "homogenous50")
            #     self.plotted = True

        self.selection_method.select(self.population)

        gen_stats.calculate_stats_after_selection()
        self.run_stats.update_stats_for_generation(gen_stats, self.gen_i)

        self.genetic_operator.apply(self.population)
        # Update ids
        self.population.generate_new_ids()

        return gen_stats

    def __calculate_final_stats(self, run_i):
        # if run_i < RUNS_TO_PLOT and self.gen_i < DISTRIBUTIONS_TO_PLOT:
        has_operator = check_has_operator(self.param_names[2])
        can_plot = not has_operator and run_i < RUNS_TO_PLOT or has_operator and run_i < RUNS_TO_PLOT_WITH_OP
        if can_plot:
            self.plotter.plot_generation_stats(self.population, self.param_names, run_i, self.gen_i, "end")
            excel.write_population(self.population, self.param_names, run_i, self.gen_i, "end")

        gen_stats = GenerationStats(self.population, self.param_names)
        if can_plot:
            self.gen_stats_list.append(gen_stats)

        gen_stats.calculate_stats_before_selection(self.prev_gen_stats)
        # gen_stats.calculate_stats_after_selection()
        self.run_stats.update_final_stats(gen_stats, self.gen_i)

        return gen_stats

    def __check_success(self, gen_stats: GenerationStats):
        is_with_operator = check_has_operator(self.param_names[2])
        if self.param_names[0] == 'FconstALL' and not is_with_operator:
            return self.has_converged
        if self.param_names[0] == 'FconstALL' and is_with_operator:
            count_dict = {}
            for chr in self.population.chromosomes:
                genotype_str = convert_genotype_format(chr.genotype)
                count_dict[genotype_str] = count_dict.get(genotype_str, 0) + 1
            for key in count_dict.keys():
                percent = count_dict[key] / N
                if 0.9 > percent > 0.1:
                    return False
            return self.has_converged

        elif self.param_names[0] == 'FH' and not is_with_operator:
            return self.has_converged and gen_stats.optimal_count == N
        elif self.param_names[0] == 'FH' and is_with_operator:
            return self.has_converged and gen_stats.optimal_count >= N * 0.9
        else:
            return self.has_converged and self.population.found_close_to_optimal()

def convert_genotype_format(genotype):
    return ''.join(geno.decode('utf-8') for geno in genotype)
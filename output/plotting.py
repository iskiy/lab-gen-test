from collections import Counter

import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt
from config import N, OUTPUT_FOLDER
import os
from model.population import Population
from stats.generation_stats import GenerationStats
import numpy as np
from matplotlib.ticker import MaxNLocator


class StatPlotter:
    def __init__(self, dpi=70):
        self.dpi = dpi
        self.fig, self.ax = plt.subplots(dpi=self.dpi)

    def plot_run_stats(self,
                       gen_stats_list: list[GenerationStats],
                       param_names: tuple[str],
                       run_i):
        reproduction_rates = [gen_stats.reproduction_rate for gen_stats in gen_stats_list if
                              gen_stats.reproduction_rate is not None]
        losses_of_diversity = [gen_stats.loss_of_diversity for gen_stats in gen_stats_list if
                               gen_stats.loss_of_diversity is not None]
        self.__plot_stat2(reproduction_rates, losses_of_diversity, param_names, run_i, 'Reproduction Rate',
                          'Loss of Diversity',
                          'rr_and_lod', 1, 0., 1.)

        unique_Xs = [gen_stats.unique_X for gen_stats in gen_stats_list]
        self.__plot_stat(unique_Xs, param_names, run_i, 'Number of Unique Chromosomes', 'unique_count')

        if param_names[0] != 'FconstALL':
            f_avgs = [gen_stats.f_avg for gen_stats in gen_stats_list]
            self.__plot_stat(f_avgs, param_names, run_i, 'Fitness Average', 'f_avg')

            f_bests = [gen_stats.f_best for gen_stats in gen_stats_list]
            self.__plot_stat(f_bests, param_names, run_i, 'Highest Fitness', 'f_best')

            intensities = [gen_stats.intensity for gen_stats in gen_stats_list if gen_stats.intensity is not None]
            self.__plot_stat(intensities, param_names, run_i, 'Selection Intensity', 'intensity', 1)

            differences = [gen_stats.difference for gen_stats in gen_stats_list if gen_stats.difference is not None]
            self.__plot_stat(differences, param_names, run_i, 'Selection Difference', 'difference', 1)

            self.__plot_stat2(differences, intensities, param_names, run_i, 'Difference', 'Intensity',
                              'intensity_and_difference', 1)

            f_stds = [gen_stats.f_std for gen_stats in gen_stats_list]
            self.__plot_stat(f_stds, param_names, run_i, 'Fitness Standard Deviation', 'f_std')

            optimal_counts = [gen_stats.optimal_count for gen_stats in gen_stats_list]
            self.__plot_stat(optimal_counts, param_names, run_i, 'Number of Optimal Chromosomes', 'optimal_count')

            num_of_bests = [gen_stats.num_of_best for gen_stats in gen_stats_list]
            self.__plot_stat(num_of_bests, param_names, run_i, 'Number of Best Chromosomes', 'best_count')

            Prs = [gen_stats.Pr for gen_stats in gen_stats_list]
            self.__plot_stat(Prs, param_names, run_i, 'Pressure (f_best/f_avg)', 'pressure')

            Pfets = [gen_stats.Pfet for gen_stats in gen_stats_list]
            self.__plot_stat(Pfets, param_names, run_i, 'Fisher’s Exact Test', 'fisher', 1)

            Ptaus = [gen_stats.Ptau for gen_stats in gen_stats_list]
            self.__plot_stat(Ptaus, param_names, run_i, 'Kendall’s τ-b', 'kendall', 1, -1., 1.)

            Pr1 = [gen_stats.Pr1 for gen_stats in gen_stats_list]
            self.__plot_stat(Pr1, param_names, run_i, 'Pr1', 'pr1', 0)

            growth_rates = [gen_stats.growth_rate for gen_stats in gen_stats_list]
            if len(growth_rates) > 0:
                growth_rates = growth_rates[1:]
            self.__plot_stat(growth_rates, param_names, run_i, 'Growth Rate', 'growth_rate', 1)

    def plot_generation_stats(self,
                              population: Population,
                              param_names: tuple[str],
                              run_i, gen_i, reason):
        # self.__plot_genotype_distribution(population, param_names, run_i, gen_i, reason)
        self.__plot_genotype_distribution2(population, param_names, run_i, gen_i, reason)
        if param_names[0] != 'FconstALL':
            self.__plot_fitness_distribution(population, param_names, run_i, gen_i, reason)
        if param_names[0] not in ['FconstALL', 'FH']:
            self.__plot_phenotype_distribution(population, param_names, run_i, gen_i, reason)

    def __plot_stat(self,
                    data,
                    param_names: tuple[str],
                    run_i,
                    set_ylabel,
                    file_name, start_index=0, y_min=None, y_max=None):
        param_hierarchy = self.__get_path_hierarchy(param_names, run_i)
        path = '/'.join(param_hierarchy)

        if not os.path.exists(path):
            os.makedirs(path)

        self.ax.clear()
        x_values = range(start_index, len(data) + start_index)
        self.ax.plot(x_values, data)
        self.ax.set_ylabel(set_ylabel)
        self.ax.set_xlabel('Generation')

        self.ax.xaxis.set_major_locator(MaxNLocator(integer=True))

        if y_min is None:
            self.ax.autoscale(enable=True, axis='y')
        else:
            self.ax.set_ylim(y_min, y_max)

        self.fig.savefig(f'{path}/{file_name}.png', dpi=self.dpi)

        txt_path = f'{path}/{file_name}.txt'
        with open(txt_path, 'w') as txt_file:
            txt_file.write("x\ty\n")
            for x, y in zip(x_values, data):
                txt_file.write(f"{x}\t{y}\n")

    def __plot_stat2(self,
                     data1, data2,
                     param_names: tuple[str],
                     run_i,
                     label1, label2,
                     file_name, start_index=0, y_min=None, y_max=None):
        param_hierarchy = self.__get_path_hierarchy(param_names, run_i)
        path = '/'.join(param_hierarchy)

        if not os.path.exists(path):
            os.makedirs(path)

        self.ax.clear()
        x_values1 = range(start_index, len(data1) + start_index)
        self.ax.plot(x_values1, data1, label=label1)
        self.ax.plot(x_values1, data2, label=label2)

        self.ax.set_xlabel('Generation')

        self.ax.xaxis.set_major_locator(MaxNLocator(integer=True))

        if y_min is None:
            self.ax.autoscale(enable=True, axis='y')
        else:
            self.ax.set_ylim(y_min, y_max)

        self.ax.legend()
        self.fig.savefig(f'{path}/{file_name}.png', dpi=self.dpi)

        txt_path = f'{path}/{file_name}.txt'
        with open(txt_path, 'w') as txt_file:
            # Define column widths for alignment
            column_widths = {'x': 6, 'label1': 20, 'label2': 20}

            # Write headers
            txt_file.write(
                f"{'x'.ljust(column_widths['x'])}\t{label1.ljust(column_widths['label1'])}\t{label2.ljust(column_widths['label2'])}\n")

            # Write data
            for x, y1, y2 in zip(x_values1, data1, data2):
                txt_file.write(
                    f"{str(x).ljust(column_widths['x'])}\t{str(y1).ljust(column_widths['label1'])}\t{str(y2).ljust(column_widths['label2'])}\n")

    def __plot_fitness_distribution(self,
                                    population: Population,
                                    param_names: tuple[str],
                                    run_i, gen_i, reason):
        param_hierarchy = self.__get_path_hierarchy(param_names, run_i) + ['fitness']
        path = '/'.join(param_hierarchy)

        if not os.path.exists(path):
            os.makedirs(path)

        x_max = population.fitness_function.get_optimal().fitness
        x_step = x_max / 25
        (x, y) = self.__get_distribution(population.fitnesses, x_max=x_max, x_step=x_step)
        self.ax.clear()
        self.ax.bar(x, y, width=x_step * 1.5)
        self.ax.set_xlabel('Chromosome fitness')
        self.ax.set_ylabel('Number of chromosomes')
        self.fig.savefig(f'{path}/{gen_i}-{reason}.png', dpi=self.dpi)

        txt_path = f'{path}/{gen_i}-{reason}.txt'
        with open(txt_path, 'w') as txt_file:
            # Define column headers
            headers = ['Chromosome fitness', 'Number of chromosomes']

            # Calculate column widths for alignment
            column_widths = [max(len(header), 20) for header in headers]

            # Write headers
            txt_file.write(f"{headers[0].ljust(column_widths[0])}\t{headers[1].ljust(column_widths[1])}\n")

            # Write data with aligned columns
            for x_val, y_val in zip(x, y):
                txt_file.write(f"{str(x_val).ljust(column_widths[0])}\t{str(y_val).ljust(column_widths[1])}\n")
        # plt.close()

    def __plot_phenotype_distribution(self,
                                      population: Population,
                                      param_names: tuple[str],
                                      run_i, gen_i, reason):
        param_hierarchy = self.__get_path_hierarchy(param_names, run_i) + ['phenotype']
        path = '/'.join(param_hierarchy)

        if not os.path.exists(path):
            os.makedirs(path)

        phenotypes = [population.fitness_function.get_phenotype(geno) for geno in population.genotypes]
        encoder = population.fitness_function.encoder  # 0000111 =  7
        x_min = encoder.lower_bound
        x_max = encoder.upper_bound
        x_step = (x_max - x_min) / 25
        (x, y) = self.__get_distribution(phenotypes, x_min=x_min, x_max=x_max, x_step=x_step)
        self.ax.clear()
        self.ax.bar(x, y, width=x_step * 1.2)
        self.ax.set_xlabel('Chromosome phenotype')
        self.ax.set_ylabel('Number of chromosomes')
        self.fig.savefig(f'{path}/{gen_i}-{reason}.png', dpi=self.dpi)

        txt_path = f'{path}/{gen_i}-{reason}.txt'
        with open(txt_path, 'w') as txt_file:
            # Define column headers
            headers = ['Chromosome phenotype', 'Number of chromosomes']

            # Calculate column widths for alignment
            column_widths = [max(len(header), 20) for header in headers]

            # Write headers
            txt_file.write(f"{headers[0].ljust(column_widths[0])}\t{headers[1].ljust(column_widths[1])}\n")

            # Write data with aligned columns
            for x_val, y_val in zip(x, y):
                txt_file.write(f"{str(x_val).ljust(column_widths[0])}\t{str(y_val).ljust(column_widths[1])}\n")

    def __plot_genotype_distribution2(self,
                                     population: Population,
                                     param_names: tuple[str],
                                     run_i, gen_i, reason):
        param_hierarchy = self.__get_path_hierarchy(param_names, run_i) + ['genotype_1']
        path = '/'.join(param_hierarchy)

        if not os.path.exists(path):
            os.makedirs(path)

        ones_counts = [len([True for gene in geno if gene == b'1']) for geno in population.genotypes]
        (x, y) = self.__get_distribution(ones_counts, x_max=population.fitness_function.chr_length)

        self.ax.clear()
        self.ax.bar(x, y)
        self.ax.set_xlabel('Number of 1s in genotype')
        self.ax.set_ylabel('Number of chromosomes')
        self.fig.savefig(f'{path}/{gen_i}-{reason}.png', dpi=self.dpi)
        # plt.close()
        txt_path = f'{path}/{gen_i}-{reason}.txt'
        with open(txt_path, 'w') as txt_file:
            # Define column headers
            headers = ['Number of 1s in genotype', 'Number of chromosomes']

            # Calculate column widths for alignment
            column_widths = [max(len(header), 20) for header in headers]

            # Write headers
            txt_file.write(f"{headers[0].ljust(column_widths[0])}\t{headers[1].ljust(column_widths[1])}\n")

            # Write data with aligned columns
            for x_val, y_val in zip(x, y):
                txt_file.write(f"{str(x_val).ljust(column_widths[0])}\t{str(y_val).ljust(column_widths[1])}\n")

    # def __plot_genotype_distribution(self,
    #                                  population: Population,
    #                                  param_names: tuple[str],
    #                                  run_i, gen_i, reason):
    #     param_hierarchy = self.__get_path_hierarchy(param_names, run_i) + ['genotype']
    #     path = '/'.join(param_hierarchy)
    #
    #     if not os.path.exists(path):
    #         os.makedirs(path)
    #     genotypes_processed = [self.__convert_genotype_format(geno) for geno in population.genotypes]
    #
    #     genotype_counts = Counter(genotypes_processed).items()
    #     # sorted_genotypes = sorted(genotype_counts.items(), key=lambda item: item[1], reverse=True)
    #
    #     # top_genotypes = genotype_counts.most_common(5)
    #
    #     genotypes, counts = zip(*genotype_counts)
    #     # genotypes, counts = zip(*genotype_counts.items())
    #
    #     # (x, y) = self.__get_distribution(ones_counts, x_max=population.fitness_function.chr_length)
    #
    #     self.ax.clear()
    #     self.ax.bar(genotypes, counts)
    #     self.ax.set_xlabel('Genes')
    #     self.ax.set_ylabel('Number of chromosomes')
    #     plt.xticks(rotation=90, fontsize="small")
    #     self.fig.savefig(f'{path}/{gen_i}-{reason}.png', dpi=self.dpi)
    #     # plt.close()

    def __convert_genotype_format(self, genotype):
        return ''.join(geno.decode('utf-8') for geno in genotype)

    def __get_distribution(self, data, x_min=0, x_max=None, x_step=1):
        if x_max is None:
            x_max = max(data)

        x = np.arange(x_min, x_max + x_step, x_step)
        y = np.zeros_like(x)
        for val in data:
            idx = int(round((val - x_min) / x_step))
            idx = max(0, min(idx, len(x) - 1))
            y[idx] += 1

        return (x, y)

    def __get_path_hierarchy(self, param_names, run_i):
        selection_params = param_names[4]
        param = ''
        for i in selection_params.keys():
            param += i + '=' + selection_params[i]

        return [
            OUTPUT_FOLDER,
            'graphs',
            param_names[0],  # fitness function
            str(N),
            param_names[1]+param,  # selection method
            param_names[2],  # genetic operator
            param_names[3],  # init method
            str(run_i),
        ]

    def Close(self):
        plt.close(self.fig)

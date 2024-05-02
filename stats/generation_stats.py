from config import N
import math
from scipy.stats import fisher_exact
from scipy.stats import kendalltau
from collections import Counter
import numpy as np
from model.population import Population

# stats that are used for graphs
class GenerationStats:
    def __init__(self, population: Population, param_names: tuple[str]):
        self.population = population
        self.param_names = param_names

        self.f_avg = None
        self.f_std = None
        self.f_best = None
        self.num_of_best = None
        self.optimal_count = None
        self.growth_rate = None
        self.difference = None
        self.intensity = None
        self.reproduction_rate = None
        self.loss_of_diversity = None

        self.unique_X = None

        self.Pr = None
        self.Pr1 = None
        self.Pfet = None
        self.Ptau = None

    def calculate_stats_before_selection(self, prev_gen_stats):
        self.ids_before_selection = self.population.get_ids()
        self.parents = self.population

        self.unique_X = len(set([''.join(geno.decode('utf-8') for geno in ch.genotype)
                                                 for ch in set(self.population.chromosomes)]))

        if self.param_names[0] != 'FconstALL':
            self.f_avg = self.population.get_fitness_avg()
            self.f_std = calculate_standard_deviation(self.population.fitnesses)
            current_f_best = self.population.get_fitness_max()
            self.num_of_best = self.population.count_fitness_at_least(current_f_best)
            self.optimal_count = self.population.count_optimal_genotype()
            
            if not prev_gen_stats:
                pass
                # self.growth_rate = 1
            elif prev_gen_stats.f_best == current_f_best:
                # TODO check
                num_of_prev_best = self.population.count_fitness_at_least(prev_gen_stats.f_best)
                # self.population.fitness_function.get_optimal()
                self.growth_rate = num_of_prev_best / prev_gen_stats.num_of_best
            else:
                self.growth_rate = 0

            self.f_best = current_f_best

            self.Pr = self.f_best / self.f_avg

            if not prev_gen_stats:
                pass
                # self.Pr1 = 1
            else:
                self.Pr1 = self.num_of_best / prev_gen_stats.num_of_best

    def __diversity(self, ids_before_selection, ids_after_selection):
        before_count = Counter(ids_before_selection)
        after_count = Counter(ids_after_selection)

        diversity_loss = sum(min(before_count[id], after_count[id]) for id in set(ids_before_selection))
        total_before = sum(before_count.values())
        diversity_loss /= total_before

        return 1 - diversity_loss

    def __reproducibility(self, ids_before_selection, ids_after_selection):
        before_count = Counter(ids_before_selection)
        after_count = Counter(ids_after_selection)

        reproducibility_rate = sum(min(before_count[id], after_count[id]) for id in set(ids_before_selection))
        total_after = sum(after_count.values())
        reproducibility_rate /= total_after

        return reproducibility_rate

    def __intensity(self, ids_before_selection, ids_after_selection):
        before_count = Counter(ids_before_selection)
        after_count = Counter(ids_after_selection)

        unique_ids = set(ids_before_selection + ids_after_selection)
        total_intensity = sum(abs(before_count[id] - after_count[id]) for id in unique_ids)
        total_before = sum(before_count.values())
        total_intensity /= total_before

        return total_intensity

    def __growth_rate(self, ids_before_selection, ids_after_selection):
        before_count = Counter(ids_before_selection)
        after_count = Counter(ids_after_selection)

        unique_ids = set(ids_before_selection + ids_after_selection)
        total_growth = sum(max(0, after_count[id] - before_count[id]) for id in unique_ids)
        total_before = sum(before_count.values())
        growth_rate = total_growth / total_before

        return growth_rate

    def calculate_stats_after_selection(self):
        ids_after_selection = set(self.population.get_ids())
        # ids_after_selection = set(self.population.get_ids())
        # ids_before_selection = set(self.ids_before_selection)

        self.reproduction_rate = len(ids_after_selection) / N
        self.loss_of_diversity = len([True for id in self.ids_before_selection if id not in ids_after_selection]) / N
        # ids_before_selection = set(self.ids_before_selection)

        # self.reproduction_rate = self.__reproducibility(self.population.get_ids(), self.ids_before_selection)
        # self.reproduction_rate = len([True for id in self.ids_before_selection if id in ids_after_selection]) / N

        # self.loss_of_diversity = self.__diversity(self.population.get_ids(), self.ids_before_selection)
        # self.loss_of_diversity = len([True for id in self.ids_before_selection if id not in ids_after_selection]) / N

        # num_before_selection = len(ids_before_selection)
        # num_after_selection = len(ids_after_selection)
        # if num_before_selection == 0:
        #     self.loss_of_diversity = 0
        # else:
        #     loss_of_diversity = 1 - (num_after_selection / num_before_selection)
        #     self.loss_of_diversity = loss_of_diversity
        #
        # unique_parents_before_selection = len(ids_before_selection)
        # if unique_parents_before_selection == 0:
        #     self.reproduction_rate = 0
        # else:
        #     self.reproduction_rate = len(ids_after_selection) / unique_parents_before_selection

        # num_before_selection = len(ids_before_selection)
        # num_after_selection = len(ids_after_selection)
        # if num_before_selection == 0:
        #     self.loss_of_diversity = 0
        # else:
        #     loss_of_diversity = 1 - (num_after_selection / num_before_selection)
        #     self.loss_of_diversity = loss_of_diversity
        #
        # unique_parents_before_selection = len(ids_before_selection)
        # if unique_parents_before_selection == 0:
        #     self.reproduction_rate = 0
        # else:
        #     self.reproduction_rate = len(ids_after_selection) / unique_parents_before_selection

        if self.param_names[0] != 'FconstALL':
            # Fisher:
            ids_after_selection = self.population.get_ids()
            offspring_count = {}
            offspring_count_arr = []
            for id in set(self.ids_before_selection):
                count = 0
                for i in ids_after_selection:
                    if id == i:
                        count += 1
                offspring_count[id] = count
                offspring_count_arr.append(count)

            # print(self.f_avg)
            offspring_median = np.mean(offspring_count_arr) # offspring number == parent number in task
            if offspring_median == 0:
                offspring_median = 1
            A = len([True for c in self.parents.chromosomes if
                     c.fitness <= self.f_avg and offspring_count[c.id] <= offspring_median])
            B = len([True for c in self.parents.chromosomes if
                     c.fitness > self.f_avg and offspring_count[c.id] <= offspring_median])
            C = len([True for c in self.parents.chromosomes if
                     c.fitness <= self.f_avg and offspring_count[c.id] > offspring_median])
            D = len([True for c in self.parents.chromosomes if
                     c.fitness > self.f_avg and offspring_count[c.id] > offspring_median])

            table = [[A, B], [C, D]]
            # p_random = 0
            # p_random += math.comb(A + B, A) * math.comb(C + D, C) / math.comb(A+B+C+D, A+C)
            # while B != 0 and C != 0:
            #     B -= 1
            #     C -= 1
            #     A += 1
            #     D += 1
            #     p_random += math.comb(A + B, A) * math.comb(C + D, C) / math.comb(A + B + C + D, A + C)
            res = fisher_exact(table, alternative='greater')

            # self.Pfet = -math.log(p_random, 10) # in paper was lg
            self.Pfet = -np.log10(res.pvalue)

            # Kendall
            fitness = self.parents.fitnesses
            offsprings = [offspring_count[c] for c in self.parents.get_ids()]
            # print(fitness)
            # print("off", offsprings)

            concordant = 0
            discordant = 0
            ties_trait = 0
            ties_offspring = 0
            for i in range(len(offsprings)):
                for j in range(i, len(offsprings)):
                    if i != j:
                        if offsprings[i] > offsprings[j] and fitness[i] > fitness[j]:
                            concordant += 1
                        elif offsprings[i] < offsprings[j] and fitness[i] < fitness[j]:
                            concordant += 1
                        elif offsprings[i] > offsprings[j] and fitness[i] < fitness[j]:
                            discordant += 1
                        elif offsprings[i] < offsprings[j] and fitness[i] > fitness[j]:
                            discordant += 1
                        elif offsprings[i] == offsprings[j]:
                            ties_offspring += 1
                        elif fitness[i] == fitness[j]:
                            ties_trait += 1

            # tau, p_value = kendalltau(fitness, offsprings)
            n = len(self.parents.get_ids())
            m = (math.comb(n, 2))
            # print(m)
            # print(ties_trait)
            # print(ties_offspring)
            # print((m - ties_trait) * (m - ties_offspring))
            # print((concordant - discordant) / math.sqrt((m - ties_trait) * (m - ties_offspring)))
            den = (m - ties_trait) * (m - ties_offspring)
            if den != 0:
                self.Ptau = (concordant - discordant) / math.sqrt(den)
            else:
                self.Ptau = 0

        # Intensity and difference
        self.ids_before_selection = None

        if self.param_names[0] != 'FconstALL':
            self.difference = self.population.get_fitness_avg() - self.f_avg
            # cal_avg = {calculate_avg(self.population)}
            avg = self.population.get_fitness_avg()
            # print(f'calculate_avg: {cal_avg}')
            # print(f'avg: {avg}')
            if self.f_std == 0:
                self.intensity = 1
            else:
                self.intensity = self.difference / self.f_std


def calculate_avg(population):
    ids_after_selection = set(population.get_ids())
    chromosomes = population.chromosomes
    fitnesses = []
    for id in ids_after_selection:
            for chr in chromosomes:
                if chr.id == id:
                    fitness = chr.fitness
                    fitnesses.append(fitness)
                    break

    return sum(fitnesses) / len(fitnesses)


def calculate_standard_deviation(fitness_values):
    n = len(fitness_values)

    # Calculate mean fitness
    mean_fitness = sum(fitness_values) / n

    # Calculate variance
    variance = sum((x - mean_fitness) ** 2 for x in fitness_values) / n

    # Calculate standard deviation
    standard_deviation = math.sqrt(variance)

    return standard_deviation

if __name__ == '__main__':
    #
    print("Fisher")
    fitness = [0, 1, 1, 2, 3, 4, 5, 5, 7, 9]
    parents = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    childs = [3, 5, 5, 6, 8, 8, 9, 9, 10, 10]
    offspring_count = {}
    offspring_count_arr = []
    for id in parents:
        count = 0
        for i in childs:
            if id == i:
                count += 1
        offspring_count[id] = count
        offspring_count_arr.append(count)

    # print(self.f_avg)
    offspring_median = np.mean(offspring_count_arr)  # offspring number == parent number in task
    f_median = np.mean(fitness)
    if offspring_median == 0:
        offspring_median = 1
    A = len([True for c in range(len(parents)) if
             fitness[c] <= f_median and offspring_count[parents[c]] <= offspring_median])
    B = len([True for c in range(len(parents)) if
             fitness[c] > f_median and offspring_count[parents[c]] <= offspring_median])
    C = len([True for c in range(len(parents)) if
             fitness[c] <= f_median and offspring_count[parents[c]] > offspring_median])
    D = len([True for c in range(len(parents)) if
             fitness[c] > f_median and offspring_count[parents[c]] > offspring_median])

    table = [[A, B], [C, D]]
    print("A = ", A)
    print("B = ", B)
    print("C = ", C)
    print("D = ", D)

    res = fisher_exact(table, alternative='greater')
    print("P_rand = ", res.pvalue)
    print("Pfet = ", -np.log10(res.pvalue))

    print("Kendal")

    offsprings = [offspring_count[c] for c in parents]
    # print(offsprings)
    # print(fitness)
    # print("off", offsprings)

    concordant = 0
    discordant = 0
    ties_trait = 0
    ties_offspring = 0
    for i in range(len(offsprings)):
        for j in range(i, len(offsprings)):
            if i != j:
                if offsprings[i] > offsprings[j] and fitness[i] > fitness[j]:
                    concordant += 1
                elif offsprings[i] < offsprings[j] and fitness[i] < fitness[j]:
                    concordant += 1
                elif offsprings[i] > offsprings[j] and fitness[i] < fitness[j]:
                    discordant += 1
                elif offsprings[i] < offsprings[j] and fitness[i] > fitness[j]:
                    discordant += 1
                elif offsprings[i] == offsprings[j]:
                    ties_offspring += 1
                elif fitness[i] == fitness[j]:
                    ties_trait += 1

    print("Concordant = ", concordant)
    print("Discordant = ", discordant)
    print("Ties trait = ", ties_trait)
    print("Ties_offspring = ", ties_offspring)

    # tau, p_value = kendalltau(fitness, offsprings)
    n = len(parents)
    m = (math.comb(n, 2))
    # print(m)
    # print(ties_trait)
    # print(ties_offspring)
    # print((m - ties_trait) * (m - ties_offspring))
    # print((concordant - discordant) / math.sqrt((m - ties_trait) * (m - ties_offspring)))
    den = (m - ties_trait) * (m - ties_offspring)
    if den != 0:
        Ptau = (concordant - discordant) / math.sqrt(den)
    else:
        Ptau = 0

    print("Ptau = ", Ptau)

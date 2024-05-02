from config import G, N
from model.population import Population
import numpy as np
from selection.selection_method import SelectionMethod
from copy import copy
from scipy.stats import rankdata


class SUS(SelectionMethod):
    def select(self, population: Population):
        fitness_sum = 0
        fitness_scale = []

        for index, chromosome in enumerate(population.chromosomes):
            fitness_sum += chromosome.fitness
            if index == 0:
                fitness_scale.append(chromosome.fitness)
            else:
                fitness_scale.append(chromosome.fitness + fitness_scale[index - 1])

        if fitness_sum == 0:
            fitness_sum = 0.0001 * N
            fitness_scale = [0.0001 * (i + 1) for i in range(N)]

        mating_pool = self.basic_sus(population, fitness_sum, fitness_scale)
        population.update_chromosomes(mating_pool)

    @staticmethod
    def basic_sus(population: Population, fitness_sum, fitness_scale):
        mating_pool = np.empty(N, dtype=object)
        fitness_step = fitness_sum / N
        random_offset = np.random.uniform(0, fitness_step)
        current_fitness_pointer = random_offset
        last_fitness_scale_position = 0

        for i in range(N):
            for fitness_scale_position in range(last_fitness_scale_position, len(fitness_scale)):
                if fitness_scale[fitness_scale_position] >= current_fitness_pointer:
                    mating_pool[i] = copy(population.chromosomes[fitness_scale_position])
                    last_fitness_scale_position = fitness_scale_position
                    break
            current_fitness_pointer += fitness_step

        return mating_pool



class LinearRankingSUS(SelectionMethod):
    def __init__(self, b=1.6):
        self.b = b

    def select(self, population: Population):
        fitness_list = copy(population.fitnesses)
        chromosomes = copy(population.chromosomes)
        # rng = np.random.default_rng()
        # indices = np.arange(len(fitness_list))
        # rng.shuffle(indices)

        # shuffled_fitnesses = fitness_list[indices]

        length = len(fitness_list)
        ranks = np.argsort(np.argsort(fitness_list))
        probabilities = (2 - self.b) / length + 2 * ranks * (self.b - 1) / (length * (length - 1))

        cumulative_probabilities = np.cumsum(probabilities)

        step = 1 / N
        start = np.random.uniform(0, step)
        points = start + step * np.arange(N)

        selected_indices = []
        cum_prob_index = 0

        for point in points:
            while cumulative_probabilities[cum_prob_index] < point:
                cum_prob_index += 1
            if cum_prob_index >= len(cumulative_probabilities):
                cum_prob_index = len(cumulative_probabilities) - 1
            selected_indices.append(cum_prob_index)

        # original_indices = indices[selected_indices]
        selected_chromosomes = chromosomes[selected_indices]

        mating_pool = np.array([copy(chr) for chr in selected_chromosomes])
        population.update_chromosomes(mating_pool)

# class LinearRankingSUS(SelectionMethod):
#     def __init__(self, b=1.4):
#         self.b = b
#
#     def select(self, population: Population):
#         fitness_list = population.fitnesses
#
#         rng = np.random.default_rng()
#         indices = np.arange(len(fitness_list))
#         rng.shuffle(indices)
#
#         shuffled_fitnesses = fitness_list[indices]
#
#         length = len(fitness_list)
#         ranks = np.argsort(np.argsort(shuffled_fitnesses))
#         probabilities = (2 - self.b) / length + 2 * ranks * (self.b - 1) / (length * (length - 1))
#
#         cumulative_probabilities = np.cumsum(probabilities)
#
#         step = 1 / N
#         start = np.random.uniform(0, step)
#         points = start + step * np.arange(N)
#
#         selected_indices = []
#         cum_prob_index = 0
#
#         for point in points:
#             while cumulative_probabilities[cum_prob_index] < point:
#                 cum_prob_index += 1
#             if cum_prob_index >= len(cumulative_probabilities):
#                 cum_prob_index = len(cumulative_probabilities) - 1
#             selected_indices.append(cum_prob_index)
#
#         original_indices = indices[selected_indices]
#         selected_chromosomes = population.chromosomes[original_indices]
#
#         mating_pool = np.array([copy(chr) for chr in selected_chromosomes])
#         population.update_chromosomes(mating_pool)
#


class LinearModifiedRankingSUS(SelectionMethod):
    def __init__(self, b=1.4):
        self.b = b

    def select(self, population: Population):
        fitness_list = population.fitnesses

        rng = np.random.default_rng()
        indices = np.arange(len(fitness_list))
        rng.shuffle(indices)

        shuffled_fitnesses = fitness_list[indices]

        length = len(fitness_list)
        ranks = rankdata(shuffled_fitnesses, method='average') - 1
        probabilities = (2 - self.b) / length + 2 * ranks * (self.b - 1) / (length * (length - 1))

        cumulative_probabilities = np.cumsum(probabilities)

        step = 1 / N
        start = np.random.uniform(0, step)
        points = start + step * np.arange(N)

        selected_indices = []
        cum_prob_index = 0

        for point in points:
            while cumulative_probabilities[cum_prob_index] < point:
                cum_prob_index += 1
            if cum_prob_index >= len(cumulative_probabilities):
                cum_prob_index = len(cumulative_probabilities) - 1
            selected_indices.append(cum_prob_index)

        original_indices = indices[selected_indices]
        selected_chromosomes = population.chromosomes[original_indices]

        mating_pool = np.array([copy(chr) for chr in selected_chromosomes])
        population.update_chromosomes(mating_pool)


class DisruptiveSUS(SelectionMethod):
    def select(self, population: Population):
        fitness_sum = 0
        fitness_scale = []
        f_avg = population.get_fitness_avg()

        for index, chromosome in enumerate(population.chromosomes):
            f_scaled = abs(chromosome.fitness - f_avg)
            fitness_sum += f_scaled
            if index == 0:
                fitness_scale.append(f_scaled)
            else:
                fitness_scale.append(f_scaled + fitness_scale[index - 1])

        if fitness_sum == 0:
            fitness_sum = 0.0001 * N
            fitness_scale = [0.0001 * (i + 1) for i in range(N)]

        mating_pool = SUS.basic_sus(population, fitness_sum, fitness_scale)
        population.update_chromosomes(mating_pool)


class BlendedSUS(SelectionMethod):
    def __init__(self):
        self.i = 0

    def select(self, population: Population):
        fitness_sum = 0
        fitness_scale = []

        for index, chromosome in enumerate(population.chromosomes):
            f_scaled = chromosome.fitness / (G + 1 - self.i)
            fitness_sum += f_scaled
            if index == 0:
                fitness_scale.append(f_scaled)
            else:
                fitness_scale.append(f_scaled + fitness_scale[index - 1])

        if fitness_sum == 0:
            fitness_sum = 0.0001 * N
            fitness_scale = [0.0001 * (i + 1) for i in range(N)]

        mating_pool = SUS.basic_sus(population, fitness_sum, fitness_scale)
        population.update_chromosomes(mating_pool)

        self.i += 1


class WindowSUS(SelectionMethod):
    def __init__(self, h=2):
        self.h = h
        self.f_h_worst = []

    def select(self, population: Population):
        if len(self.f_h_worst) < self.h:
            self.f_h_worst.append(min(population.fitnesses))
        else:
            self.f_h_worst[0] = self.f_h_worst[1]
            self.f_h_worst[1] = min(population.fitnesses)
        f_worst = min(self.f_h_worst)

        fitness_sum = 0
        fitness_scale = []

        for index, chromosome in enumerate(population.chromosomes):
            f_scaled = chromosome.fitness - f_worst
            fitness_sum += f_scaled
            if index == 0:
                fitness_scale.append(f_scaled)
            else:
                fitness_scale.append(f_scaled + fitness_scale[index - 1])

        if fitness_sum == 0:
            fitness_sum = 0.0001 * N
            fitness_scale = [0.0001 * (i + 1) for i in range(N)]

        mating_pool = SUS.basic_sus(population, fitness_sum, fitness_scale)
        population.update_chromosomes(mating_pool)

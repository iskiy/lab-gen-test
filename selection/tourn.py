from config import N
from model.encoding import BinaryEncoderUni
from model.fitness_functions import Fx2, FH
from model.population import Population
import numpy as np
from selection.selection_method import SelectionMethod
import copy as cc
import random

from config import G, N
from model.chromosome import Chromosome
from model.population import Population
import numpy as np
from selection.selection_method import SelectionMethod
from typing import Callable
from copy import deepcopy


class TS(SelectionMethod):
    def __init__(self, t: int = 2, p: Callable = lambda x, fits: 1, replacement: bool = False):
        super().__init__()
        self.t = t
        self.p = p
        self.replacement = replacement

    def select(self, population: Population):
        if self.replacement:
            num_offsprings, mating_pool = self.tournament_with_replacement(population)
        else:
            num_offsprings, mating_pool = self.prob_tournament_without_replacement(population)

        mating_pool = np.array([cc.copy(chr) for chr in mating_pool])
        np.random.shuffle(mating_pool)
        population.update_chromosomes(mating_pool)
        #
        # zeros = len([True for i in num_offsprings if i == 0])
        # unique = len(set(population.get_ids()))
        # print(zeros, unique)
        # print(num_offsprings)

        return num_offsprings

    def prob_tournament_without_replacement(self, population: Population):
        np.random.shuffle(population.chromosomes)
        num_offsprings = [0 for _ in range(len(population.chromosomes))]
        # num_offsprings_w_l = [[0, 0] for _ in range(len(population.chromosomes))]
        mating_pool = []
        remains = []
        # count_w = 0
        # count_t = 0
        for _ in range(self.t):
            copy_population = np.array([cc.deepcopy(ch) for ch in population.chromosomes])
            chromosomes = np.array([(i, ch) for i, ch in enumerate(copy_population)])
            np.random.shuffle(chromosomes)
            while len(chromosomes) > 0:
                if len(chromosomes) < self.t:
                    remains.extend(chromosomes)
                    break
                np.random.shuffle(chromosomes)
                indexes = list(range(len(chromosomes)))
                np.random.shuffle(indexes)
                contestants_indexes = np.random.choice(indexes, size=self.t, replace=False)
                contestants = chromosomes[contestants_indexes]
                np.random.shuffle(contestants)

                chromosomes = chromosomes[np.setdiff1d(indexes, contestants_indexes)]

                fitnesses = [ch[1].fitness for ch in contestants]
                s_conts_fits = sorted(list(zip(contestants, fitnesses)), key=lambda x: x[1], reverse=True)
                probs = [self.p(s_conts_fits[i][1], fitnesses) * (1 - self.p(s_conts_fits[i][1], fitnesses)) ** i for i
                         in range(len(s_conts_fits))]
                probs[-1] = 1 - sum(probs[:-1])
                # print(r)
                combined_data_l = list(zip(s_conts_fits, probs))
                np.random.shuffle(combined_data_l)

                chrs_l = [chr for chr, _ in combined_data_l]
                probabilities_l = [probabilities for _, probabilities in combined_data_l]

                winner = chrs_l[np.random.choice(len(chrs_l), p=probabilities_l)]
                num_offsprings[winner[0][0]] += 1
                mating_pool.append(cc.deepcopy(winner[0][1]))

        if remains:
            for _ in range(len(remains) // self.t):
                np.random.shuffle(remains)
                # indexes = list(range(len(remains)))
                # np.random.shuffle(indexes)
                # contestants_indexes = np.random.choice(indexes, size=self.t, replace=False)
                # contestants = remains[contestants_indexes]
                # np.random.shuffle(remains)
                #
                # remains = remains[np.setdiff1d(indexes, contestants_indexes)]
                contestants = remains[:self.t]
                remains = remains[self.t:]

                fitnesses = [ch[1].fitness for ch in contestants]
                s_conts_fits = sorted(list(zip(contestants, fitnesses)), key=lambda x: x[1], reverse=True)
                probs = [self.p(s_conts_fits[i][1], fitnesses) * (1 - self.p(s_conts_fits[i][1], fitnesses)) ** i for i
                         in range(len(s_conts_fits))]
                probs[-1] = 1 - sum(probs[:-1])
                combined_data_l = list(zip(s_conts_fits, probs))
                np.random.shuffle(combined_data_l)

                chrs_l = [chr for chr, _ in combined_data_l]
                probabilities_l = [probabilities for _, probabilities in combined_data_l]

                winner = chrs_l[np.random.choice(len(chrs_l), p=probabilities_l)]
                num_offsprings[winner[0][0]] += 1
                mating_pool.append(cc.deepcopy(winner[0][1]))
        # print("w/l", count_w, count_t)
        # print(num_offsprings)
        # print(num_offsprings_w_l)
        return num_offsprings, mating_pool

    def tournament_without_replacement(self, population: Population):
        np.random.shuffle(population.chromosomes)
        num_offsprings = [0 for _ in range(len(population.chromosomes))]
        mating_pool = []
        remains = []
        for _ in range(self.t):
            copy_population = np.array([cc.deepcopy(ch) for ch in population.chromosomes])
            chromosomes = np.array([(i, ch) for i, ch in enumerate(copy_population)])
            np.random.shuffle(chromosomes)
            while len(chromosomes) > 0:
                if len(chromosomes) < self.t:
                    remains.extend(chromosomes)
                    break
                contestants = chromosomes[:self.t]
                chromosomes = chromosomes[self.t:]

                fitnesses = [ch[1].fitness for ch in contestants]
                s_conts_fits = sorted(list(zip(contestants, fitnesses)), key=lambda x: x[1], reverse=True)
                probs = [self.p(s_conts_fits[i][1], fitnesses) * (1 - self.p(s_conts_fits[i][1], fitnesses)) ** i for i
                         in range(len(s_conts_fits))]
                probs[-1] = 1 - sum(probs[:-1])
                winner = s_conts_fits[np.random.choice(len(s_conts_fits), p=probs)]
                num_offsprings[winner[0][0]] += 1
                mating_pool.append(cc.deepcopy(winner[0][1]))

        if remains:
            for _ in range(len(remains) // self.t):
                contestants = remains[:self.t]
                remains = remains[self.t:]

                fitnesses = [ch[1].fitness for ch in contestants]
                s_conts_fits = sorted(list(zip(contestants, fitnesses)), key=lambda x: x[1], reverse=True)
                probs = [self.p(s_conts_fits[i][1], fitnesses) * (1 - self.p(s_conts_fits[i][1], fitnesses)) ** i for i
                         in range(len(s_conts_fits))]
                probs[-1] = 1 - sum(probs[:-1])
                winner = s_conts_fits[np.random.choice(len(s_conts_fits), p=probs)]
                num_offsprings[winner[0][0]] += 1
                mating_pool.append(cc.deepcopy(winner[0][1]))

        return num_offsprings, mating_pool

    def tournament_with_replacement(self, population: Population):
        num_offsprings = [0 for _ in range(len(population.chromosomes))]
        mating_pool = []

        copy_population = deepcopy(population)
        chromosomes = [(i, ch) for i, ch in enumerate(copy_population.chromosomes)]
        np.random.shuffle(chromosomes)
        while len(mating_pool) < N:
            contestants = [chromosomes[np.random.randint(0, len(chromosomes))] for _ in range(self.t)]
            fitnesses = [ch[1].fitness for ch in contestants]
            s_conts_fits = sorted(list(zip(contestants, fitnesses)), key=lambda x: x[1], reverse=True)
            probs = [self.p(s_conts_fits[i][1], fitnesses) * (1 - self.p(s_conts_fits[i][1], fitnesses)) ** i for i in
                     range(len(s_conts_fits))]
            probs[-1] = 1 - sum(probs[:-1])
            winner = s_conts_fits[np.random.choice(len(s_conts_fits), p=probs)]
            num_offsprings[winner[0][0]] += 1
            mating_pool.append(winner[0][1])

        return num_offsprings, mating_pool

class TournamentWithReplacement(SelectionMethod):
    def __init__(self, t=2, p=1):
        self.t = t
        self.p = p

    def select(self, population: Population):
        chromosomes = list(population.chromosomes)
        # Shuffle the chromosomes to ensure randomness
        rng = np.random.default_rng()
        rng.shuffle(chromosomes)
        mating_pool = []

        for _ in range(N):
            # np.random.shuffle(chromosomes)
            # Randomly select t chromosomes for the tournament, with replacement
            chosen = rng.choice(chromosomes, size=self.t, replace=True)

            # Calculate fitness for each selected chromosome
            healths = [chrom.fitness for chrom in chosen]

            # Pair up chromosomes and their fitness
            combined_data = list(zip(chosen, healths))

            # Sort the pairs based on fitness
            sorted_data = sorted(combined_data, key=lambda x: x[1], reverse=True)

            # Calculate probabilities based on ranking
            probabilities = [self.p * ((1 - self.p) ** i) for i in range(self.t)]
            probabilities[-1] = 1 - sum(probabilities[:-1])

            sorted_chr = [chr for chr, health in sorted_data]

            # combined_data_l = list(zip(sorted_chr, probabilities))
            # rng.shuffle(combined_data_l)
            #
            # chrs_l = [chr for chr, _ in combined_data_l]
            # probabilities_l = [probabilities for _, probabilities in combined_data_l]
            # print(probabilities)
            chromosome = rng.choice(sorted_chr, p=probabilities)

            # Select a winner with the probability distribution
            # winner = np.random.choice([chrom for chrom, _ in combined_data], p=probabilities)
            mating_pool.append(cc.deepcopy(chromosome))

        # Update the population with the selected chromosomes
        population.update_chromosomes(np.array(mating_pool))

def p10(x, fits):
    return 1

def p08(x, fits):
    return 0.8

def p075(x, fits):
    return 0.75

def p07(x, fits):
    return 0.7

def p06(x, fits):
    return 0.6

def pxi(x, fits):
    return x/np.sum(fits) if np.sum(fits) > 0 else 0

def p05(x, fits):
    return 0.5


class TournamentWithoutReplacement(SelectionMethod):
    def __init__(self, t=2, p=1):
        self.t = t
        self.p = p

    def __select(self, population: Population):
        chromosomes = list(population.chromosomes)
        population_copies = [cc.deepcopy(chromosomes) for i in range(self.t)]
        # population_copies = [copy.deepcopy(chromosomes) for i in range(self.t)]

        # print("pp", len(population_copies))

        mating_pool = []

        reminder_array = []
        for copy in population_copies:
            np.random.shuffle(copy)
            # print("c", copy)
            for i in range(len(chromosomes) // self.t + 1):
                np.random.shuffle(copy)
                if len(copy) < self.t:
                    reminder_array += copy
                else:

                    chosen = copy[:self.t]
                    healths = [chrom.fitness for chrom in chosen]

                    # print(chosen)
                    copy = copy[self.t:]

                    combined_data = list(zip(chosen, healths))

                    sorted_data = sorted(combined_data, key=lambda x: x[1], reverse=True)
                    # print(sorted_data)

                    sorted_chr = [chr for chr, health in sorted_data]

                    probabilities = [self.p * (1 - self.p) ** i for i in range(len(sorted_chr))]
                    probabilities[-1] = 1 - sum(probabilities[:-1])

                    # probabilities /= np.sum(probabilities)

                    # print(probabilities)

                    # combined_data_l = list(zip(sorted_chr, probabilities))
                    # np.random.shuffle(combined_data_l)
                    #
                    # chrs_l = [chr for chr, _ in combined_data_l]
                    # probabilities_l = [probabilities for _, probabilities in combined_data_l]
                    # # print(probabilities)
                    # chromosome = np.random.choice(chrs_l, p=probabilities_l)
                    chromosome = np.random.choice(sorted_chr, p=probabilities)

                    # chromosome = sorted_chr[chosen_index]
                    # print(probabilities)

                    # chromosome = np.random.choice(sorted_chr, p=probabilities)

                    mating_pool.append(chromosome)

        if len(reminder_array):
            for i in range(len(reminder_array) // self.t + 1):
                if (len(reminder_array) > self.t):
                    chosen = reminder_array[:self.t]
                else:
                    chosen = reminder_array
                if len(chosen) == 0:
                    break
                healths = [chrom.fitness for chrom in chosen]

                reminder_array = reminder_array[self.t:]

                combined_data = list(zip(chosen, healths))

                sorted_data = sorted(combined_data, key=lambda x: x[1], reverse=True)

                sorted_chr = [chr for chr, health in sorted_data]

                probabilities = [self.p * (1 - self.p) ** i for i in range(len(sorted_chr))]

                probabilities[-1] = 1 - sum(probabilities[:-1])
                # print(probabilities)

                # combined_data_l = list(zip(sorted_chr, probabilities))
                # np.random.shuffle(combined_data_l)
                #
                # chrs_l = [chr for chr, _ in combined_data_l]
                # probabilities_l = [probabilities for _, probabilities in combined_data_l]
                #
                chromosome = np.random.choice(sorted_chr, p=probabilities)

                # chromosome = sorted_chr[chosen_index]

                # chromosome = np.random.choice(sorted_chr, p=probabilities)

                mating_pool.append(chromosome)

        mating_pool = np.array(mating_pool)
        # print(mating_pool)
        population.update_chromosomes(mating_pool)

    def select(self, population: Population):
        chromosomes = np.array(population.chromosomes)
        population_copies = [cc.deepcopy(chromosomes) for i in range(self.t)]
        # population_copies = [copy.deepcopy(chromosomes) for i in range(self.t)]

        # print("pp", len(population_copies))
        rng = np.random.default_rng()

        mating_pool = []

        reminder_array = np.array([])
        for copy in population_copies:
            super_copy = cc.deepcopy(copy)
            rng.shuffle(super_copy)
            # print("c", copy)
            for i in range(len(chromosomes) // self.t + 1):
                if len(super_copy) < self.t:
                    reminder_array = np.concatenate((reminder_array, super_copy))
                else:
                    indexes = range(len(chromosomes))
                    contestants_indexes = np.random.choice(indexes, size=self.t, replace=False)
                    chosen = chromosomes[contestants_indexes]

                    super_copy = chromosomes[np.setdiff1d(indexes, contestants_indexes)]
                    healths = [chrom.fitness for chrom in chosen]

                    # print(chosen)

                    combined_data = list(zip(chosen, healths))

                    sorted_data = sorted(combined_data, key=lambda x: x[1], reverse=True)
                    # print(sorted_data)

                    sorted_chr = [chr for chr, health in sorted_data]

                    probabilities = [self.p * (1 - self.p) ** i for i in range(self.t)]
                    probabilities[-1] = 1 - sum(probabilities[:-1])

                    # probabilities /= np.sum(probabilities)

                    # print(probabilities)

                    combined_data_l = list(zip(sorted_chr, probabilities))
                    rng.shuffle(combined_data_l)

                    chrs_l = [cc.deepcopy(chr) for chr, _ in combined_data_l]
                    probabilities_l = [probabilities for _, probabilities in combined_data_l]
                    # print(probabilities)
                    # f = rng.random()
                    # acc = 0
                    # index = 0
                    # for p in range(len(probabilities_l)):
                    #     if f < acc:
                    #         index = p - 1
                    #     else:
                    #         acc += probabilities_l[p]
                    # chromosome = chrs_l[index]

                    chromosome = np.random.choice(chrs_l, p=probabilities_l)

                    # chromosome = sorted_chr[chosen_index]
                    # print(probabilities)

                    # chromosome = np.random.choice(sorted_chr, p=probabilities)

                    mating_pool.append(cc.deepcopy(chromosome))

        if len(reminder_array):
            for i in range(len(reminder_array) // self.t + 1):
                if (len(reminder_array) > self.t):
                    chosen = reminder_array[:self.t]
                else:
                    chosen = reminder_array
                if len(chosen) == 0:
                    break
                healths = [chrom.fitness for chrom in chosen]

                reminder_array = reminder_array[self.t:]

                combined_data = list(zip(chosen, healths))

                sorted_data = sorted(combined_data, key=lambda x: x[1], reverse=True)

                sorted_chr = [chr for chr, health in sorted_data]

                probabilities = [self.p * (1 - self.p) ** i for i in range(len(sorted_chr))]

                probabilities[-1] = 1 - sum(probabilities[-1])
                # print(probabilities)

                combined_data_l = list(zip(sorted_chr, probabilities))
                np.random.shuffle(combined_data_l)

                chrs_l = [chr for chr, _ in combined_data_l]
                probabilities_l = [probabilities for _, probabilities in combined_data_l]
                #
                chromosome = np.random.choice(chrs_l, p=probabilities_l)

                # chromosome = sorted_chr[chosen_index]

                # chromosome = np.random.choice(sorted_chr, p=probabilities)

                mating_pool.append(cc.deepcopy(chromosome))

        mating_pool = np.array(mating_pool)
        # print(mating_pool)
        population.update_chromosomes(mating_pool)

    # def select(self, population):
    #     chromosomes = np.array(population.chromosomes)
    #     population_copies = [cc.deepcopy(chromosomes) for i in range(self.t)]
    #     np.random.shuffle(population_copies)
    #
    #     mating_pool = []
    #
    #     for copy in population_copies:
    #         while len(copy) >= self.t:
    #             chosen = np.random.choice(copy, size=self.t, replace=False)
    #             copy = np.setdiff1d(copy, chosen)
    #
    #             healths = [chrom.fitness for chrom in chosen]
    #             combined_data = list(zip(chosen, healths))
    #             sorted_data = sorted(combined_data, key=lambda x: x[1], reverse=True)
    #             sorted_chr = [chr for chr, health in sorted_data]
    #
    #             probabilities = [self.p * (1 - self.p) ** i for i in range(self.t)]
    #             probabilities /= np.sum(probabilities)
    #
    #             chromosome = np.random.choice(sorted_chr, p=probabilities)
    #             mating_pool.append(cc.deepcopy(chromosome))
    #
    #     mating_pool = np.array(mating_pool)
    #     population.update_chromosomes(mating_pool)
    #
    def select(self, population: Population):
        selected_indices = []
        bad_indices = []

        while len(selected_indices) < len(population.chromosomes):
            # If there are not enough good indices left, reset bad indices
            if len(population.chromosomes) - len(bad_indices) < self.t:
                bad_indices = []

            # Select t random indices not in the bad indices list
            tour_indexes = np.random.choice(
                np.setdiff1d(range(0, len(population.chromosomes)), bad_indices),
                self.t,
                replace=False
            )

            # Choose the winner index based on the highest fitness
            winner = tour_indexes[np.argmax(population.fitnesses[tour_indexes])]

            # Add the winner index to the selected indices list
            selected_indices.append(winner)

            # Extend the bad indices list with the tour indexes
            bad_indices.extend(tour_indexes)

        # Convert the selected indices list to a numpy array
        selected_indices = np.array(selected_indices)

        # Update the population chromosomes with the selected indices
        new_population = population.update_chromosomes(
            cc.deepcopy(population.chromosomes[selected_indices])
        )


if __name__ == '__main__':
    # copy = np.array([1, 1, 1, 2])
    # chosen = np.random.choice(copy, size=2, replace=False)
    #
    # # Print the original arrays
    # print("Original copy array:", copy)
    # print("Elements to remove:", chosen)
    #
    # # Remove the elements present in 'chosen' from 'copy'
    # for elem in chosen:
    #     index = np.where(copy == elem)[0]
    #     copy = np.delete(copy, index)
    # print(copy)

    binar = BinaryEncoderUni(4, 0.0, 0.15, 0.01)
    #print(binar.get_all_values())
    b = TS(2, p075, False)
    population = Population(FH(100))
    p = []
    for i in range(20):
        #print(population)
        x = b.select(population)
        zeros = len([True for i in x if i == 0])
        p.append(zeros)
    p = np.array(p)
    print(np.mean(p))
    # b = TS(2, p10, False)
    # population = Population(FH(100))
    # for i in range(20):
    #     #print(population)
    #     b.select(population)

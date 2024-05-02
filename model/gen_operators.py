from model.population import Population
from config import N, get_p_m
import random
import numpy as np
from model.chromosome import Chromosome


class GeneticOperator:
    @staticmethod
    def apply(population: Population):
        raise NotImplementedError()


class BlankGenOperator(GeneticOperator):
    @staticmethod
    def apply(population):
        for chr_i in range(N):
            population.chromosomes[chr_i].id = chr_i

class Crossover(GeneticOperator):
    @staticmethod
    def apply(population: Population):
        np.random.shuffle(population.chromosomes)
        children = np.empty(N, dtype=object)
        l = population.fitness_function.chr_length

        for i in range(N // 2):
            chr_are_equal = True
            for j in range(l):
                if population.chromosomes[i*2].genotype[j] != population.chromosomes[i*2+1].genotype[j]:
                    chr_are_equal = False
                    break

            if chr_are_equal:
                children[i*2] = population.chromosomes[i*2]
                children[i*2+1] = population.chromosomes[i*2+1]
                children[i*2].id = i*2
                children[i*2+1].id = i*2+1
            else:
                crossing_point = np.random.randint(1, l)

                genotype1 = np.concatenate([
                    population.chromosomes[i*2].genotype[:crossing_point],
                    population.chromosomes[i*2+1].genotype[crossing_point:]
                ])
                genotype2 = np.concatenate([
                    population.chromosomes[i*2+1].genotype[:crossing_point],
                    population.chromosomes[i*2].genotype[crossing_point:]
                ])
                
                children[i*2] = Chromosome(i*2, genotype1, population.fitness_function)
                children[i*2+1] = Chromosome(i*2+1, genotype2, population.fitness_function)
        population.update_chromosomes(children)


class Mutation(GeneticOperator):
    @staticmethod
    def apply(population: Population):
        l = population.fitness_function.chr_length
        p_m = get_p_m(l)
        
        for chromosome in population.chromosomes:
            for bit_i in range(l):
                if random.random() < p_m:
                    chromosome.genotype[bit_i] = int(not chromosome.genotype[bit_i])
                    chromosome.update_fitness()

        for chr_i in range(N):
            population.chromosomes[chr_i].id = chr_i
            
        population.update()


class CrossoverAndMutation(GeneticOperator):
    @staticmethod
    def apply(population: Population):
        np.random.shuffle(population.chromosomes)
        children = np.empty(N, dtype=object)
        l = population.fitness_function.chr_length
        p_m = get_p_m(l)

        for i in range(N // 2):
            chr_are_equal = True
            for j in range(l):
                if population.chromosomes[i*2].genotype[j] != population.chromosomes[i*2+1].genotype[j]:
                    chr_are_equal = False
                    break

            if chr_are_equal:
                children[i*2] = population.chromosomes[i*2]
                children[i*2+1] = population.chromosomes[i*2+1]
                children[i*2].id = i*2
                children[i*2+1].id = i*2+1
            else:
                crossing_point = np.random.randint(1, l)

                genotype1 = np.concatenate([
                    population.chromosomes[i*2].genotype[:crossing_point],
                    population.chromosomes[i*2+1].genotype[crossing_point:]
                ])
                genotype2 = np.concatenate([
                    population.chromosomes[i*2+1].genotype[:crossing_point],
                    population.chromosomes[i*2].genotype[crossing_point:]
                ])
                
                children[i*2] = Chromosome(i*2, genotype1, population.fitness_function)
                children[i*2+1] = Chromosome(i*2+1, genotype2, population.fitness_function)
        
        for chromosome in children:
            for bit_i in range(l):
                if random.random() < p_m:
                    chromosome.genotype[bit_i] = int(not chromosome.genotype[bit_i])
                    chromosome.update_fitness()
        population.update_chromosomes(children)
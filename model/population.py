import numpy as np
from config import N, EPS, N_LAST_GENS
from model.chromosome import Chromosome
from copy import deepcopy, copy
from scipy.stats import binom


class Population:
    def __init__(self, fitness_function, seed=0, optimal_initialise_per=0, optimal_initialise_percent=0,
                 with_operator=0, can_have_optimal=False, chromosomes=None):
        self.fitness_function = fitness_function
        self.optimal_initialise_per = optimal_initialise_per
        self.optimal_initialise_percent = optimal_initialise_percent
        self.with_operator = with_operator

        if chromosomes is not None:
            self.chromosomes = chromosomes
        else:
            used = 0
            self.chromosomes = np.empty(N, dtype=object)
            if optimal_initialise_per != 0:
                used = int(optimal_initialise_per)
                for i in range(int(optimal_initialise_per)):
                    self.chromosomes[i] = copy(fitness_function.get_optimal())
            elif optimal_initialise_percent != 0:
                used = int(optimal_initialise_percent * N)
                for i in range(used):
                    self.chromosomes[i] = copy(fitness_function.get_optimal())
            # rng = np.random.default_rng(seed=seed)
            rng = np.random.default_rng()
            for chr_i in range(used, N):
                # print(used)
                # genotype = rng.choice([b'0', b'1'], fitness_function.chr_length)
                genotype = generate_binomial_scipy(fitness_function.chr_length, 0.5)
                # if with_operator:
                if can_have_optimal:
                    # print(genotype.tobytes(), fitness_function.get_optimal().genotype.tobytes())
                    while genotype.tobytes() == fitness_function.get_optimal().genotype.tobytes():
                        # print("ppp")
                        genotype = rng.choice([b'0', b'1'], fitness_function.chr_length)
                self.chromosomes[chr_i] = Chromosome(chr_i, genotype, fitness_function)

        self.generate_new_ids()

        # print("------")
        # for i in self.chromosomes:
        #     print(i)
        # TODO save started population and graph
        self.update()
        # self.calculate_distribution_statistics()

    def has_converged(self, f_avgs, param_names):

        isHasOperator = check_has_operator(param_names[2])

        if not isHasOperator:
            return self.is_homogenous_100()

        return self.is_homogenous_99()
        # if param_names[0] == 'FconstALL':
        #     return self.is_homogenous_99()

        # return self.has_f_avg_converged(f_avgs)

    def has_f_avg_converged(self, f_avgs):
        if len(f_avgs) < N_LAST_GENS:
            return False

        diffs = []
        for i in range(1, len(f_avgs)):
            curr = f_avgs[i]
            prev = f_avgs[i - 1]
            diffs.append(abs(curr - prev))

        return all(x <= EPS for x in diffs)

    def is_homogenous_percent(self, percent):
        l = self.fitness_function.chr_length
        for i in range(l):
            n_zeros = len([True for g in self.genotypes if g[i] == b'0'])
            percentage = n_zeros / N
            if 1 - percent < percentage < percent:
                return False
        return True

    def is_homogenous_99(self):
        l = self.fitness_function.chr_length
        for i in range(l):
            n_zeros = len([True for g in self.genotypes if g[i] == b'0'])
            percentage = n_zeros / N
            if 0.01 < percentage < 0.99:
                return False
        return True

    def is_homogenous_100(self):
        etalon = self.genotypes[0]
        return all([np.array_equal(geno, etalon) for geno in self.genotypes])

    def found_close_to_optimal(self):
        for chr in self.chromosomes:
            if self.fitness_function.check_chromosome_success(chr):
                return True
        return False

    def get_fitness_max(self):
        res = np.max(self.fitnesses)
        return res

    def get_fitness_avg(self):
        return np.mean(self.fitnesses)

    def get_fitness_std(self):
        return np.std(self.fitnesses)

    def count_fitness_at_least(self, min_fitness):
        return len([True for f in self.fitnesses if f >= min_fitness])

    def count_optimal_genotype(self):
        optimal = self.fitness_function.get_optimal().genotype
        return len([True for g in self.genotypes if np.array_equal(g, optimal)])

    def get_ids(self):
        return [chr.id for chr in self.chromosomes]

    def update(self):
        self.fitnesses = np.array([chr.fitness for chr in self.chromosomes])
        self.genotypes = np.array([chr.genotype for chr in self.chromosomes])

    def update_chromosomes(self, chromosomes):
        self.chromosomes = chromosomes
        self.update()

    def update_chromosomes_id(self):
        """ update chromosome ids in such way that equal chromosomes have equal ids """
        id_map = {}
        for i, chr in enumerate(self.chromosomes):
            gen = self.__convert_genotype_format(chr.genotype)
            if gen in id_map:
                old_id = copy(self.chromosomes[i].id)
                self.chromosomes[i].id = id_map[gen]
                # print(f'id: {i}, old_id: {old_id}, gen {gen}')
            else:
                id_map[gen] = i

    def calculate_distribution_statistics(self):
        fitness_values = self.fitnesses
        mean = np.mean(fitness_values)
        std_dev = np.std(fitness_values)
        range_value = np.ptp(fitness_values)  # Peak-to-peak (max - min) range
        print(f"Mean Fitness: {mean:.4f}, Standard Deviation: {std_dev:.4f}, Range: {range_value}")
        return range_value

    def generate_new_ids(self):
        for i, chr in enumerate(self.chromosomes):
            chr.id = i

    def __convert_genotype_format(self, genotype):
        return ''.join(geno.decode('utf-8') for geno in genotype)

    def __deepcopy__(self, memo):
        return Population(self.fitness_function, chromosomes=deepcopy(self.chromosomes))

    def __str__(self):
        return str(np.array([str(chr) for chr in self.chromosomes]))


def check_has_operator(param):
    return has_mutation(param) or has_crossover(param)


def has_mutation(param):
    return 'mut' in param


def has_crossover(param):
    return 'over' in param


# def generate_genotype(length, prob):
#     # Generate a binomial distribution with 0s and 1s
#     binomial_values = np.random.binomial(1, prob, length)
#     # Convert the binomial values to bytes ('b0' and 'b1')
#     return np.array([b'1' if val else b'0' for val in binomial_values])

def generate_binomial_scipy(length, prob):
    # Generate random variates from a binom distribution
    binomial_values = binom.rvs(1, prob, size=length)

    # Convert to binary characters 'b0' and 'b1'
    return np.array([b'1' if val else b'0' for val in binomial_values])


def generate_genotype(length, prob):
    # Generate a binomial distribution with 0s and 1s
    count_ones = np.random.binomial(length, prob)
    genotype = [b'1'] * count_ones + [b'0'] * (length - count_ones)

    # Shuffle to mix the '0's and '1's
    np.random.shuffle(genotype)
    return np.array(genotype)

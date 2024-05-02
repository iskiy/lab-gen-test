import numpy as np
from config import DELTA, SIGMA, get_pop_seed
from model.chromosome import Chromosome
from model.population import Population
from model.encoding import Encoder
import math


class FitnessFunc:
    def __init__(self, chr_length):
        self.chr_length = chr_length
        self.optimal = None

    def apply(self, genotype):
        raise NotImplementedError()

    def get_optimal(self):
        raise NotImplementedError()

    def get_phenotype(self, genotype):
        raise NotImplementedError()

    def generate_population_for_run(self, run_i):
        return Population(self, seed=get_pop_seed(run_i))

    def check_chromosome_success(self, chr: Chromosome):
        y_diff = abs(chr.fitness - self.get_optimal().fitness)
        x_diff = abs(self.get_phenotype(chr.genotype) - self.get_phenotype(self.get_optimal().genotype))
        return y_diff <= DELTA and x_diff <= SIGMA


# -	FconstALL(X) визначена на ланцюжках заданої довжини, причому FconstALL(X)=100 для всіх ланцюжків;
# [Функція використовується тільки для дослідження шуму відбору та врати різноманітності]
class FconstALL(FitnessFunc):
    def apply(self, genotype):
        return 100

    def get_optimal(self):
        if not self.optimal:
            self.optimal = Chromosome(0, np.full(self.chr_length, b'0'), self)
        return self.optimal

    def get_phenotype(self, genotype):
        return 0

    def check_chromosome_success(self, ch):
        return True


# useless
class FHD(FitnessFunc):
    def __init__(self, delta, chr_length):
        super().__init__(chr_length)
        self.delta = delta

    def apply(self, genotype):
        k = len([True for gene in genotype if gene == b'0'])
        return (self.chr_length - k) + k * self.delta

    def get_optimal(self):
        if not self.optimal:
            self.optimal = Chromosome(0, np.full(self.chr_length, b'0'), self)
        return self.optimal

    def get_phenotype(self, genotype):
        return len([True for gene in genotype if gene == b'1'])


# 	FH(X)=(l-k) – модифікована для максимізації функція, що обчислює відстань Геммінга до оптимального ланцюжка
# 	Xopt=«0...0», де k – кількість «1» в ланцюжку; очевидно, FH(«0...0»)=l. [Фактично – кількість «0» в ланцюжку]
class FH(FitnessFunc):
    def __init__(self, chr_length):
        super().__init__(chr_length)

    def apply(self, genotype):
        k = len([True for gene in genotype if gene == b'1'])
        return self.chr_length - k

    def get_optimal(self):
        if not self.optimal:
            self.optimal = Chromosome(0, np.full(self.chr_length, b'0'), self)
        return self.optimal

    def get_phenotype(self, genotype):
        return len([True for gene in genotype if gene == b'0'])


#	f(x)  степенева (розглянути такі випадки):
#   y=x^2, 0≤x≤10.23 . Глобальний максимум:  y=〖(10.23)〗^2," " x=10.23.
class Fx2(FitnessFunc):
    def __init__(self, encoder: Encoder):
        super().__init__(encoder.length)
        self.encoder = encoder
        self.is_caching = encoder.length <= 12
        self.cache_dict = {}
        if self.is_caching:
            for v in self.encoder.get_all_values():
                self.cache_dict[v.tobytes()] = self.encoder.decode(v) ** 2

    def apply(self, genotype):
        if self.is_caching:
            # print(int(genotype.tobytes()))
            return self.cache_dict[genotype.tobytes()]
        return self.encoder.decode(genotype) ** 2

    def get_optimal(self):
        if not self.optimal:
            optimal_genotype = self.encoder.encode(max(abs(self.encoder.upper_bound), abs(self.encoder.lower_bound)))
            self.optimal = Chromosome(0, optimal_genotype, self)
        return self.optimal

    def get_phenotype(self, chromosome):
        return self.encoder.decode(chromosome)


#   y =〖(5.12)〗 ^ 2 - x ^ 2, -5.12≤x < 5.12
#   Глобальний максимум: y =〖(5.12)〗 ^ 2, x = 0.
class F5122subx2(FitnessFunc):
    def __init__(self, encoder: Encoder):
        super().__init__(encoder.length)
        self.encoder = encoder
        self.is_caching = encoder.length <= 12
        self.cache_dict = {}
        if self.is_caching:
            for v in self.encoder.get_all_values():
                self.cache_dict[v.tobytes()] = 5.12 ** 2 - self.encoder.decode(v) ** 2

    def apply(self, genotype):
        if self.is_caching:
            return self.cache_dict[genotype.tobytes()]
        return 5.12 ** 2 - self.encoder.decode(genotype) ** 2

    def get_optimal(self):
        if not self.optimal:
            optimal_genotype = self.encoder.encode(0)
            self.optimal = Chromosome(0, optimal_genotype, self)
        return self.optimal

    def get_phenotype(self, genotype):
        return self.encoder.decode(genotype)


# 	f(x)  експоненційна виду e^(c*x)  (розглянути випадки):
# 	c=0.25,
# 	c=1,
# 	c=2.
# 0≤x≤10.23. Глобальний максимум:  y=e^(c*x),x=10.23.
class Fexp(FitnessFunc):
    def __init__(self, c, encoder: Encoder):
        self.c = c
        super().__init__(encoder.length)
        self.encoder = encoder
        self.is_caching = encoder.length <= 12
        self.cache_dict = {}
        if self.is_caching:
            for v in self.encoder.get_all_values():
                self.cache_dict[v.tobytes()] = math.exp(self.c * self.encoder.decode(v))

    def apply(self, genotype):
        if self.is_caching:
            return self.cache_dict[genotype.tobytes()]
        return math.exp(self.c * self.encoder.decode(genotype))

    def get_optimal(self):
        if not self.optimal:
            optimal_genotype = self.encoder.encode(self.encoder.upper_bound)
            self.optimal = Chromosome(0, optimal_genotype, self)
        return self.optimal

    def get_phenotype(self, chromosome):
        return self.encoder.decode(chromosome)


#   Растригіна
#   f(x)=|10cos (2πa)-a^2 |+10cos (2πx)  -x^2,a=7;
#   -5.12≤x<5.12 Глобальний максимум:  y=49,x=0.
class FRastrigina(FitnessFunc):
    def __init__(self, a, encoder: Encoder):
        self.a = a
        super().__init__(encoder.length)
        self.encoder = encoder
        self.is_caching = encoder.length <= 12
        self.cache_dict = {}
        if self.is_caching:
            for v in self.encoder.get_all_values():
                x = self.encoder.decode(v)
                self.cache_dict[v.tobytes()] = abs(10 * math.cos(2 * math.pi * self.a) - self.a ** 2) + \
                                               10 * math.cos(2 * math.pi * x) - x ** 2

    def apply(self, genotype):
        if self.is_caching:
            return self.cache_dict[genotype.tobytes()]
        x = self.encoder.decode(genotype)
        return abs(10 * math.cos(2 * math.pi * self.a) - self.a ** 2) \
            + 10 * math.cos(2 * math.pi * x) - x ** 2

    def get_optimal(self):
        if not self.optimal:
            optimal_genotype = self.encoder.encode(0)
            self.optimal = Chromosome(0, optimal_genotype, self)
        return self.optimal

    def get_phenotype(self, chromosome):
        return self.encoder.decode(chromosome)


#	Функція Деба 2 (Decreasing maxima, Deb’s test function 2):
#   Глобальний максимум: x=0.1
class FDeba2(FitnessFunc):
    def __init__(self, encoder: Encoder):
        super().__init__(encoder.length)
        self.encoder = encoder
        self.is_caching = encoder.length <= 12
        self.cache_dict = {}
        if self.is_caching:
            for v in self.encoder.get_all_values():
                x = self.encoder.decode(v)
                self.cache_dict[v.tobytes()] = math.exp(-2 * math.log(2) * ((x - 0.1)/0.8)**2)\
                                               * math.sin(5 * math.pi * x) ** 6

    def apply(self, genotype):
        if self.is_caching:
            return self.cache_dict[genotype.tobytes()]
        x = self.encoder.decode(genotype)
        return math.exp(-2 * math.log(2) * ((x - 0.1)/0.8)**2) * math.sin(5 * math.pi * x) ** 6

    def get_optimal(self):
        if not self.optimal:
            optimal_genotype = self.encoder.encode(0.1)
            self.optimal = Chromosome(0, optimal_genotype, self)
        return self.optimal

    def get_phenotype(self, chromosome):
        return self.encoder.decode(chromosome)


#	Функція Деба 4 (Uneven decreasing maxima, Deb’s test function 4):
#   Глобальний максимум: x≈0.080
class FDeba4(FitnessFunc):
    def __init__(self, encoder: Encoder):
        super().__init__(encoder.length)
        self.encoder = encoder
        self.is_caching = encoder.length <= 12
        self.cache_dict = {}
        if self.is_caching:
            for v in self.encoder.get_all_values():
                x = self.encoder.decode(v)
                self.cache_dict[v.tobytes()] = math.exp(-2 * math.log(2) * ((x - 0.08)/0.854)**2)\
                                               * math.sin(5 * math.pi * (x**0.75 - 0.05)) ** 6

    def apply(self, genotype):
        if self.is_caching:
            return self.cache_dict[genotype.tobytes()]
        x = self.encoder.decode(genotype)
        return math.exp(-2 * math.log(2) * ((x - 0.08)/0.854)**2) * math.sin(5 * math.pi * (x**0.75 - 0.05)) ** 6

    def get_optimal(self):
        if not self.optimal:
            optimal_genotype = self.encoder.encode(0.08)
            self.optimal = Chromosome(0, optimal_genotype, self)
        return self.optimal

    def get_phenotype(self, chromosome):
        return self.encoder.decode(chromosome)


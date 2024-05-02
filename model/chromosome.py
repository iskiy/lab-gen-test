class Chromosome:
    def __init__(self, id, genotype, fitness_function):
        self.id = id
        self.fitness_function = fitness_function
        self.set_genotype(genotype)

    def set_genotype(self, genotype):
        self.genotype = genotype
        self.update_fitness()

    def update_fitness(self):
        self.fitness = self.fitness_function.apply(self.genotype)

    def __lt__(self, other):
        return self.fitness < other.fitness

    def __copy__(self):
        return Chromosome(self.id, self.genotype.copy(), self.fitness_function)

    def __deepcopy__(self, memo):
        return self.__copy__()

    def __str__(self):
        return f"Chr{self.id}({str(self.genotype.tobytes().decode('utf-8'))})"

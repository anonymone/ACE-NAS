import individual


class population():
    def __init__(self, config, arg=None):
        self.popSize = int(config['population setting']['popSize'])
        self.generation = dict()
        # initiate generation
        self.generation['0'] = [individual.SEA_individual(
            config['individual setting']) for x in range(self.popSize)]

    def get_population(self, index=-1):
        if index == -1:
            index = str(len(self.generation)-1)
        return self.generation[index]

    def add_population(self, newPop):
        if len(newPop) != self.popSize:
            print('The new population size is not suit rule')
        self.generation[str(len(self.generation))] = newPop

    def update_population(self, newPop, index = -1):
        if index == -1:
            index = str(len(self.generation)-1)
        self.generation[index] = newPop


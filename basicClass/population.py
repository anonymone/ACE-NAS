import individual

class population():
    def __init__(self, config, arg = None):
        self.pop = dict(
            zip([str(x) for x in range(int(config['population setting']['popSize']))], \
            [individual.SEA_individual(config=config['individual setting']) for x in range(int(config['population setting']['popSize']))]) \
            )
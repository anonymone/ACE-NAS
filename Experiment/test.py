import configparser
import sys
sys.path.append('./')
sys.path.append('./Population/')
sys.path.append('./Engine/')

config = configparser.ConfigParser()
config.read('./Experiments/config.txt')

from individual import individual
from population import population
from StateControl import decoder

pop = population(config=config)
ind = pop.get_population()
ind = ind[0]
decoder = decoder(config['individual setting'])

decoder.get_model(ind.get_dec())
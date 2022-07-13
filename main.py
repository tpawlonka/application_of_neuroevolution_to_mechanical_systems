import argparse
import datetime
import json
import os

import jsonpickle

import concrete_evolution as ce
import concrete_simulation as cs
from interfaces.evolution import Population, Individual
from interfaces.simulation import SimParams

parser = argparse.ArgumentParser(description='')
parser.add_argument('-d', type=str, default='', help="results directory")
parser.add_argument('-nr', type=int, default=0, help="no render")
# parser.add_argument('-ls', type=str, default='', help="load individual from string")
# parser.add_argument('-st', type=str, default='', help="display statistics")
parser.add_argument('-s', type=int, default=64, help="population size")
parser.add_argument('-t', type=int, default=2500, help="simulation time")
parser.add_argument('-lp', type=str, default=None, help="load prototype")
parser.add_argument('-sp', type=float, default=132.5, help="starting point")
parser.add_argument('-cmp', type=float, default=0.05, help="compatibility threshold")
parser.add_argument('-cll', type=float, default=0.1, help="culling threshold")
parser.add_argument('-cfw', type=float, default=1, help="coefficient w")
parser.add_argument('-cfe', type=float, default=1, help="coefficient e")
parser.add_argument('-cfd', type=float, default=1, help="coefficient d")
parser.add_argument('-mch', type=int, default=10, help="mutation chance %")
parser.add_argument('-mrl', type=int, default=5, help="mutation rolls")
parser.add_argument('-mmc', type=int, default=5, help="mutation maximum change %")
parser.add_argument('-pm', type=int, default=0, help="save picture mode")
parser.add_argument('-lb', type=str, default='', help="path to save file from which to load the best specimen as prototype")
args = parser.parse_args()
if __name__ == '__main__':
    Individual.prototype_fitness_tester = ce.CFitnessTester()
    results_dir = args.d + datetime.datetime.now().__str__().replace('.', '').replace(':', '').replace(' ', '')
    params = SimParams(gravity=(0.0, -900), time=args.t, save_picture=args.pm, save_directory=results_dir)
    ce.SimData.middle = args.sp
    sim_data = ce.SimData(0, 0, 0, False, 0)
    if args.nr == 0:
        sim = cs.CSimulation(sim_params=params)
    if args.nr == 1:
        sim = cs.CSimulationSilent(sim_params=params)

    mutator = ce.CMutator(args.mch, args.mrl, args.mmc)
    selector = ce.CSelector(compatibility_threshold=args.cmp,
                            culling_threshold=args.cll,
                            coefficient_w=args.cfw,
                            coefficient_e=args.cfe,
                            coefficient_d=args.cfd,
                            starting_pop_number=args.s)
    crossover = ce.CCrossover()
    pop = Population(mutator, selector, crossover, li=list(), timeout=None)
    if args.lp is not None:
        initializer = ce.PreloaderGenomeInitializer(args.lp)
    else:
        initializer = ce.CGenomeInitializer()
    # if args.ls != '':
    #     loaded = json.loads(args.ls)
    #     pop.members = [Individual(ce.CEncoderDecoder(), loaded)]
    if args.lb != '':
        with open(args.lb) as file:
            data = file.read()
        parsed_json = json.loads(data)
        specimens = parsed_json[0:len(parsed_json) - 2]
        sort = sorted(specimens, key=lambda x: x['fitness'], reverse=True)
        print(sort[0]['genome'])
        initializer = ce.PreloaderGenomeInitializer(sort[0]['genome'])
    os.mkdir(results_dir)
    pop.initialize(codec=ce.CEncoderDecoder(), gen_init=initializer, starting_pop_number=args.s)
    generation_number = 2
    while True:
        pop.map_fitness(sim)
        savedata = list()
        pop.members.sort(key=lambda x: x.fitness())
        avg_fitness = 0
        species = dict()

        top_fitness = 0
        for member in pop.members:
            savedata.append({
                'fitness': member.fitness(),
                'mean_x': member.sim_data.mean_x,
                'species': member.species,
                'genome': member.genome,
                'guid': member.guid
            })
            avg_fitness += member.fitness()
            if member.fitness() > top_fitness:
                top_fitness = member.fitness()
            species[member.species] = 1
        avg_fitness = avg_fitness / len(pop.members)
        print("Average fitness in generation "+pop.generation_number.__str__()+": "+avg_fitness.__str__())
        print("Top fitness in generation "+pop.generation_number.__str__()+": "+top_fitness.__str__())
        savedata.append({
            'avg_fitness': avg_fitness,
            'top_fitness': top_fitness,
            'Number of species': selector.last_species_alive_count,
            'SimParams': jsonpickle.encode(params),
            'CSelector': jsonpickle.encode(selector),
            'CMutator': jsonpickle.encode(mutator),
            'CCrossover': jsonpickle.encode(crossover),
            'top_guid_bnode': ce.BNode.highest_guid,
            'top_guid_nnnode': ce.NNNode.highest_guid,
            'top_innov_bconn': ce.BConnection.global_innovation_counter,
            'top_innov_nnconn': ce.NNConnection.global_innovation_counter,
            # 'compatibility_threshold': ce.CSelector.compatibility_threshold
        })
        file = open(results_dir + "/" + pop.generation_number.__str__() + ".json", "w")
        file.write(json.dumps(savedata))
        pop = pop.new_generation()
        pop.generation_number = generation_number
        generation_number += 1

    # loop this until some break condition

#  DONE: write loop detection for the NN in the sim (used exception catching on breaching recursion limit instead,
#        should actually be more performant this way)
#  TODO: ensure removing connections cannot leave orphaned nodes
#  TODO: ensure connections between two inputs cannot happen
#  TODO: make sure input nodes can be added
#  TODO: add sensors for actuator length (maybe unnecessary)
#  TODO: see if you can replace the recursion with iteration
#  TODO: add online stats graph drawing when running, and stats graph drawing from save file
#  TODO: add Yifan Hu's graph drawing algorithm for the networks
#  TODO: add ability to load from file

#  DONE: tweak fitness function to penalize complex bodies less DONE
#  DONE: move logic out of main MOSTLY DONE
#  DONE: preserve species over generations DONE

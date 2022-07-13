from __future__ import annotations
import copy
import datetime
import json
import os
from abc import abstractmethod, ABC
from dataclasses import dataclass
import concurrent.futures
from time import sleep
from typing import Optional

import numpy as np

from interfaces.simulation import Simulation, SimParams, SimData


class AbstractBodyPhenotype(ABC):
    @abstractmethod
    def get(self) -> list:
        pass


class AbstractMindPhenotype(ABC):
    @abstractmethod
    def get(self) -> list:
        pass

    def get_layers(self) -> list[set]:
        pass

    def get_matrices(self, layers: list[set]) -> list:
        pass


@dataclass
class Phenotype:
    body: AbstractBodyPhenotype
    mind: AbstractMindPhenotype


class EncoderDecoder(ABC):
    @abstractmethod
    def encode(self, graph) -> dict:
        pass

    @abstractmethod
    def decode(self, sequence) -> Phenotype:
        pass
        # methods encode and decode which convert a binary sequence to and from a graph representation


class FitnessTester(ABC):
    @abstractmethod
    def test(self, individual: Individual):
        return individual.sim_data.mean_x


class Individual:
    genome: dict
    codec: EncoderDecoder
    species: int
    species_counter = 1
    sim_data: SimData
    fitness_tester: FitnessTester
    prototype_fitness_tester = None
    newly_speciated = False
    generation: int
    guid: int
    guid_counter = 0
    adjuster: int

    def __init__(self, codec: EncoderDecoder, genome):
        self.genome = genome
        self.codec = codec
        self.species = 0
        self.generation = 1
        self.fitness_tester = Individual.prototype_fitness_tester
        self.guid = Individual.guid_counter
        self.adjuster = 1
        Individual.guid_counter += 1
    # takes an encoder_decoder as an argument, stores the genome as a binary sequence and as a graph
    # the simulation.py file will deal with converting the graph to box2d objects

    def decode(self):
        return self.codec.decode(self.genome)

    def encode(self, phenotype: Phenotype):
        self.genome = self.codec.encode(phenotype)

    def fitness(self):
        return self.fitness_tester.test(self)


class Mutator(ABC):
    @abstractmethod
    def mutate(self, population_members: list[Individual]) -> list[Individual]:
        pass  # returns the input population list, after performing its mutation strategy on all of them
    # mutator should be implemented with a mutation strategy


class Selector(ABC):
    @abstractmethod
    def select(self, population_members: list[Individual]) -> list[Individual]:
        pass  # list of selected members
    # selector should be implemented with a selection strategy

    def speciate(self, population_members: list[Individual]) -> list[Individual]:
        pass


class Crossover(ABC):
    @abstractmethod
    def crossover(self, population_members: list[Individual]) -> list[Individual]:
        pass  # return list of new population members created through crossover
        # between the original population members
    # crossover should be implemented with a crossing-over strategy


class GenomeInitializer(ABC):
    prototype: list

    @abstractmethod
    def create(self):
        pass


class Population:
    members: list[Individual]
    mutator: Mutator
    selector: Selector
    crossover: Crossover
    timeout: Optional[int]

    def __init__(self, mutator: Mutator, selector: Selector, crossover: Crossover, li: list[Individual], timeout=1):
        self.members = li
        self.mutator = mutator
        self.selector = selector
        self.crossover = crossover
        self.timeout = timeout
        self.generation_number = 1

    def initialize(self, codec: EncoderDecoder, starting_pop_number, gen_init: GenomeInitializer):
        gen_init.create()
        for i in range(0, starting_pop_number):
            clone = copy.deepcopy(gen_init.prototype)
            self.members.append(Individual(copy.deepcopy(codec), clone))

    # def map_fitness(self, simulation: Simulation, fitness: Fitness, sim_params: SimParams):
    #     for member in self.members:
    #         member.fitness = simulation.run_sim(sim_params, member)
    def map_fitness(self, simulation: Simulation):
        # futures = list()
        # with concurrent.futures.ProcessPoolExecutor() as executor:
        #     for member in self.members:
        #         futures.append(executor.submit(simulation.run_sim, sim_params, member.codec.decode(member.genome), fitness))
        # while True:
        #     for i in range(0, len(futures)):
        #         if not futures[i].done():
        #             break
        #         for j in range(0, len(futures)):
        #             self.members[j].fitness = futures[j].result()
        with concurrent.futures.ProcessPoolExecutor() as executor:
            # iterator = executor.map(simulation.run_sim, self.members, timeout=None, chunksize=1)
            scheduled = list()
            x = 0
            y = 200
            for member in self.members:
                scheduled.append(executor.submit(simulation.run_sim, member, (x, y)))
                x = x+600
                if x > (3000+1920):
                    x = 0
                    y += 650
                    if y > 1000:
                        y = 200
            while True:
                done = True
                for schedule in scheduled:
                    if not schedule.done():
                        done = False
                if done:
                    break
            updated_population = []
            iterator = list()
            for schedule in scheduled:
                iterator.append(schedule.result(None))
            for result in iterator:
                updated_population.append(result)

            # Cull members who for whatever reason (e.g. pygame or pymunk error) don't have SimData
            for member in updated_population:
                if not hasattr(member, 'sim_data'):
                    member.sim_data = SimData(0, 0, 0, True)
                num_elements = len(member.genome['body_nodes']) \
                               + len(member.genome['body_connections']) \
                               + len(member.genome['nn_nodes']) \
                               + len(member.genome['nn_connections'])
                if num_elements == 0:
                    member.sim_data = SimData(0, 0, 0, True)
            self.members = updated_population
            # while True:
            #     try:
            #         if iterator is None:
            #             break
            #         element = next(iterator)
            #         if element is None:
            #             continue
            #         updated_population.append(element)
            #     except StopIteration:
            #         break

    def new_generation(self):
        new_members = self.mutator.mutate(self.crossover.crossover(self.selector.select(self.members)))
        return Population(self.mutator, self.selector, self.crossover, new_members)

    def avg_fitness(self):
        i = 0
        for member in self.members:
            i += member.fitness()
        return i / len(self.members)

    def hgh_fitness(self):
        i = 0
        for member in self.members:
            if member.fitness() > i:
                i = member.fitness
        return i

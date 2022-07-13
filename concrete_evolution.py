from __future__ import annotations
import copy
import importlib
import math
import sys
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass
from typing import Optional

import numpy as np
import random

import pymunk

from interfaces.evolution import Mutator, Selector, Crossover, EncoderDecoder, \
    AbstractBodyPhenotype, Individual, GenomeInitializer, AbstractMindPhenotype, Phenotype, FitnessTester
from interfaces.simulation import SimData


class CFitnessTester(FitnessTester):
    def test(self, individual: Individual):
        raw_fit = individual.sim_data.mean_x
        # complexity penalty:
        # raw_fit * 1/elements^2
        num_elements = len(individual.genome['body_nodes']) \
                       + len(individual.genome['body_connections']) \
                       + len(individual.genome['nn_nodes']) \
                       + len(individual.genome['nn_connections'])
        # print("num elements ", num_elements)
        # outlier penalty:
        highest_x = individual.genome['body_nodes'][0]['x']
        highest_y = individual.genome['body_nodes'][0]['y']
        lowest_x = individual.genome['body_nodes'][0]['x']
        lowest_y = individual.genome['body_nodes'][0]['y']
        for node in individual.genome['body_nodes']:
            if node['x'] > highest_x:
                highest_x = node['x']
            elif node['x'] < lowest_x:
                lowest_x = node['x']
            if node['y'] > highest_y:
                highest_y = node['y']
            elif node['y'] < lowest_y:
                lowest_y = node['y']
        bounded_area = self.get_bounded_area(highest_x, lowest_x, highest_y, lowest_y)
        delta1 = abs(SimData.middle - highest_x)
        delta2 = abs(SimData.middle - lowest_x)
        normalizer = max(individual.sim_data.bounded_area, bounded_area)
        area_term = (1 - abs((individual.sim_data.bounded_area / normalizer) - (bounded_area / normalizer)))
        fitness = (((raw_fit / (num_elements/10) ** 2)
                   + (raw_fit / max(delta1, delta2))) / 2) * area_term
        return fitness / individual.adjuster

    def test_exp2(self, individual: Individual):
        raw_fit = individual.sim_data.mean_x
        # outlier penalty:
        highest_x = individual.genome['body_nodes'][0]['x']
        highest_y = individual.genome['body_nodes'][0]['y']
        lowest_x = individual.genome['body_nodes'][0]['x']
        lowest_y = individual.genome['body_nodes'][0]['y']
        for node in individual.genome['body_nodes']:
            if node['x'] > highest_x:
                highest_x = node['x']
            elif node['x'] < lowest_x:
                lowest_x = node['x']
            if node['y'] > highest_y:
                highest_y = node['y']
            elif node['y'] < lowest_y:
                lowest_y = node['y']
        bounded_area = self.get_bounded_area(highest_x, lowest_x, highest_y, lowest_y)
        delta1 = abs(SimData.middle - highest_x)
        delta2 = abs(SimData.middle - lowest_x)
        normalizer = max(individual.sim_data.bounded_area, bounded_area)
        area_term = (1 - abs((individual.sim_data.bounded_area / normalizer) - (bounded_area / normalizer)))
        fitness = ((raw_fit + (raw_fit / max(delta1, delta2))) / 2) * area_term
        return fitness / individual.adjuster

    def get_bounded_area(self, highest_x, lowest_x, highest_y, lowest_y):
        len_1 = highest_y-lowest_y
        len_2 = highest_x-lowest_x
        return len_1 * len_2


class CEncoderDecoder(EncoderDecoder):
    cache: Optional[Phenotype]

    def __init__(self):
        self.cache = None

    def decode(self, genome):
        if self.cache is not None:
            return self.cache
        body_nodes = dict()
        body_connections = list()
        neural_network_nodes = dict()
        neural_network_connections = list()
        for node in genome['body_nodes']:
            body_nodes[node['guid']] = BNode.regenerate(x=node['x'], y=node['y'], guid=node['guid'],
                                                        physical_weight=node['physical_weight'])
        for conn in genome['body_connections']:
            # if not conn['enabled']:
            #     continue
            if 'new' in conn.keys():
                manifested = BConnection.new(body_nodes[conn['in_node']], body_nodes[conn['out_node']],
                                             conn['strength'], conn['guid'])
            else:
                manifested = BConnection.regenerate(in_node=body_nodes[conn['in_node']],
                                                    out_node=body_nodes[conn['out_node']],
                                                    guid=conn['guid'],
                                                    strength=conn['strength'],
                                                    color=conn['color'],
                                                    enabled=conn['enabled'],
                                                    innovation_id=conn['innovation_id'])
            body_connections.append(manifested)
            body_nodes[conn['in_node']].outputs.append(manifested)
            body_nodes[conn['out_node']].inputs.append(manifested)
        for node in genome['nn_nodes']:
            sensor = None
            if node["sensor"] is not None:
                if node["sensor"][0] in ["SensorNodeX", "SensorNodeY"]:
                    lbl = getattr(importlib.import_module("concrete_evolution"), node["sensor"][0]).label
                    sensor = getattr(importlib.import_module("concrete_evolution"), node["sensor"][0])(
                        node=body_nodes[node["sensor"][1]],
                        sim_data=None,
                        label=lbl)
                elif node["sensor"][0] in ["SensorMeanX", "SensorMeanY"]:
                    lbl = getattr(importlib.import_module("concrete_evolution"), node["sensor"][0]).label
                    sensor = getattr(importlib.import_module("concrete_evolution"), node["sensor"][0])(node=None,
                                                                                                       sim_data=None,
                                                                                                       label=lbl)
                elif node["sensor"][0] in ["SensorClock20", "SensorClock100",
                                           "InputZeroOne", "InputPlusMinus", "SensorSignClock20"]:
                    sensor = getattr(importlib.import_module("concrete_evolution"), node["sensor"][0])(node=None,
                                                                                                       sim_data=None,
                                                                                                       label='')

            neural_network_nodes[node['guid']] = NNNode.regenerate(guid=node['guid'],
                                                                   mode=node['mode'],
                                                                   sensor=sensor)
            if node['actuator'] is not None:
                for bconn in body_connections:
                    if bconn.guid == node['actuator']:
                        neural_network_nodes[node['guid']].actuator = bconn
                        break
                if neural_network_nodes[node['guid']].actuator is None:
                    print("aha!!!")
        for conn in genome['nn_connections']:
            # if not conn['enabled']:
            #     continue
            if 'new' in conn.keys():
                manifested = NNConnection.new(conn['weight'], neural_network_nodes[conn['in_node']],
                                              neural_network_nodes[conn['out_node']], conn['guid'])
            else:
                manifested = NNConnection.regenerate(in_node=neural_network_nodes[conn['in_node']],
                                                     out_node=neural_network_nodes[conn['out_node']],
                                                     guid=conn['guid'],
                                                     enabled=conn['enabled'],
                                                     innovation_id=conn['innovation_id'],
                                                     weight=conn['weight'])
            neural_network_connections.append(manifested)
            neural_network_nodes[conn['in_node']].outputs.append(manifested)
            neural_network_nodes[conn['out_node']].inputs.append(manifested)
        body_nodes_dict = body_nodes
        body_nodes = list(body_nodes.values())
        neural_network_nodes = list(neural_network_nodes.values())
        nn_nodes_dict = neural_network_nodes
        self.cache = Phenotype(BodyPhenotype(body_nodes, body_connections),
                               MindPhenotype(neural_network_nodes, neural_network_connections))
        self.cache.body_nodes_dict = body_nodes_dict
        self.cache.nn_nodes_dict = nn_nodes_dict
        return self.cache

    def encode(self, graph: Phenotype):
        genome = {
            'body_nodes': list(),
            'body_connections': list(),
            'nn_nodes': list(),
            'nn_connections': list()
        }
        for node in graph.body.body_nodes:
            genome['body_nodes'].append({
                'x': node.x,
                'y': node.y,
                'guid': node.guid,
                'physical_weight': node.physical_weight
            })
        for node in graph.body.body_connections:
            genome['body_connections'].append({
                'in_node': node.in_node.guid,
                'out_node': node.out_node.guid,
                'enabled': node.enabled,
                'strength': node.strength,
                'color': node.color,
                'innovation_id': node.innovation_id,
                'guid': node.guid
            })
        for node in graph.mind.neural_network_nodes:
            sensor = None
            if node.sensor is not None:
                cls = type(node.sensor).__name__
                if cls in ["SensorNodeX", "SensorNodeY"]:
                    sensor = (cls, node.sensor.node.guid)
                elif cls in ["SensorMeanX", "SensorMeanY", "SensorClock20",
                             "SensorClock100", "InputZeroOne", "InputPlusMinus", "SensorSignClock20"]:
                    sensor = (cls, None)
            actuator = None
            if node.actuator is not None:
                actuator = node.actuator.guid
            genome['nn_nodes'].append({
                'mode': node.mode,
                'actuator': actuator,
                'sensor': sensor,
                'guid': node.guid
            })
        for node in graph.mind.neural_network_connections:
            genome['nn_connections'].append({
                'weight': node.weight,
                'in_node': node.in_node.guid,
                'out_node': node.out_node.guid,
                'enabled': node.enabled,
                'innovation_id': node.innovation_id,
                'guid': node.guid
            })

        return genome


class BNode:
    x: float
    y: float
    inputs: list[BConnection]
    outputs: list[BConnection]
    physical_node: pymunk.Body
    physical_shape: pymunk.Shape
    physical_connections: list[pymunk.Constraint]
    guid: int
    highest_guid = 0
    physical_weight: int

    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y
        self.inputs = []
        self.outputs = []
        self.physical_connections = []

    @classmethod
    def new(cls, x: float, y: float):
        obj = BNode(x, y)
        obj.guid = cls.highest_guid
        cls.highest_guid += 1
        obj.physical_weight = 10
        return obj

    @classmethod
    def regenerate(cls, x, y, guid, physical_weight):
        obj = BNode(x, y)
        obj.guid = guid
        obj.physical_weight = physical_weight
        return obj


class BConnection:
    in_node: BNode
    out_node: BNode
    strength: float  # 0 is a completely lose connection, limp and elastic, 1 is a completely stiff bone;
    # notionally this is a sort of weight analogous to the one in the NNConnection
    enabled: bool
    innovation_id: int
    global_innovation_counter = 0
    physical_dist: float
    color: str
    physical_spring: pymunk.DampedSpring
    guid: int
    highest_guid = 0

    def __init__(self, in_node, out_node, strength):
        self.in_node = in_node
        self.out_node = out_node
        self.strength = strength

    @classmethod
    def new(cls, in_node, out_node, strength, guid=None):
        obj = BConnection(in_node, out_node, strength)
        obj.color = "black"
        obj.enabled = True
        obj.innovation_id = cls.global_innovation_counter
        cls.global_innovation_counter += 1
        if guid is None:
            obj.guid = cls.highest_guid
            cls.highest_guid += 1
        else:
            obj.guid = guid
        return obj

    @classmethod
    def regenerate(cls, in_node, out_node, strength, guid, color, innovation_id, enabled):
        obj = BConnection(in_node, out_node, strength)
        obj.guid = guid
        obj.color = color
        obj.innovation_id = innovation_id
        obj.enabled = enabled
        return obj


@dataclass
class Sensor(ABC):
    node: Optional[BNode]
    sim_data: Optional[SimData]
    label: str

    @abstractmethod
    def get_val(self):
        pass


@dataclass
class SensorNodeX(Sensor):
    label = "X"

    def get_val(self):
        return self.node.physical_node.position.x / 10


@dataclass
class SensorNodeY(Sensor):
    label = "Y"

    def get_val(self):
        return self.node.physical_node.position.y / 10


@dataclass
class SensorTime(Sensor):
    label = "t"

    def get_val(self):
        return self.sim_data.time


@dataclass
class SensorMeanX(Sensor):
    label = "avg(X)"

    def get_val(self):
        return self.sim_data.mean_x / 10


@dataclass
class SensorMeanY(Sensor):
    label = "avg(Y)"

    def get_val(self):
        return self.sim_data.mean_y / 10


@dataclass
class SensorClock20(Sensor):
    label = 'clk20'
    val: int

    def __init__(self, node, sim_data, label):
        super(SensorClock20, self).__init__(node, sim_data, label)
        self.val = 0
        self.label = 'clk20'

    def get_val(self):
        self.val += 1
        if self.val > 20:
            self.val = 0
        return self.val


@dataclass
class SensorClock100(Sensor):
    label = 'clk100'
    val: int

    def __init__(self, node, sim_data, label):
        super(SensorClock100, self).__init__(node, sim_data, label)
        self.val = 0
        self.label = 'clk100'

    def get_val(self):
        self.val += 1
        if self.val > 100:
            self.val = 0
        return self.val


@dataclass
class InputZeroOne(Sensor):
    label = '01'
    last: bool

    def __init__(self, node, sim_data, label):
        super(InputZeroOne, self).__init__(node, sim_data, label)
        self.last = True
        self.label = '01'

    def get_val(self):
        self.last = not self.last
        return int(self.last)


@dataclass
class InputPlusMinus(Sensor):
    label = '+-'
    last: int

    def __init__(self, node, sim_data, label):
        super(InputPlusMinus, self).__init__(node, sim_data, label)
        self.last = 1
        self.label = '+-'

    def get_val(self):
        self.last *= -1
        return self.last


@dataclass
class SensorSignClock20(Sensor):
    label = "+-20"
    output: int
    clk: int

    def __init__(self, node, sim_data, label):
        super(SensorSignClock20, self).__init__(node, sim_data, label)
        self.output = 1
        self.clk = 1
        self.label = "+-20"

    def get_val(self):
        self.clk += 1
        if self.clk > 20:
            self.clk = 1
            self.output *= -1
        return self.output


class NNNode:
    mode: str  # input, output, hidden
    sensor: Optional[Sensor]  # only if mode == "input", null otherwise
    actuator: Optional[BConnection]  # only if mode == "output"
    inputs: list[NNConnection]
    outputs: list[NNConnection]
    activation_strength: float
    time_step: int
    pos: tuple[int]  # for rendering only
    guid: int
    highest_guid = 0

    def __init__(self, mode: str, sensor: Sensor = None,
                 actuator: BConnection = None):
        self.mode = mode
        self.actuator = actuator
        self.inputs = []
        self.outputs = []
        self.sensor = sensor
        self.activation_strength = -1
        # self.time_step = -1  # The simulation must set this to 0 for all mode=="input" nodes

    @classmethod
    def new(cls, mode: str, sensor: Sensor = None,
            actuator: BConnection = None):
        obj = NNNode(mode, sensor, actuator)
        obj.guid = cls.highest_guid
        cls.highest_guid += 1
        return obj

    @classmethod
    def regenerate(cls, guid: int, mode: str, sensor: Sensor = None,
                   actuator: BConnection = None):
        obj = NNNode(mode, sensor, actuator)
        obj.guid = guid
        return obj

    def activate(self) -> float:
        if self.activation_strength == -1:
            weights = []
            values = []
            if self.sensor is not None:
                self.activation_strength = self.sensor.get_val()
                return self.activation_strength
            else:
                for connection in self.inputs:
                    if not connection.enabled:
                        continue
                    weights = np.append(weights, connection.weight)
                    values = np.append(values, connection.in_node.activate())
            self.activation_strength = min(float(np.dot(weights, values)), sys.float_info.max)
            return NNNode.activation_f(self.activation_strength)
        else:
            return NNNode.activation_f(self.activation_strength)

    @staticmethod
    def activation_f(activation_strength):
        try:
            return max((1/(1+np.exp(-activation_strength))), 0.10)
        except RuntimeWarning:
            return 0


class NNConnection:
    weight: float
    in_node: NNNode
    out_node: NNNode
    enabled: bool
    innovation_id: int
    global_innovation_counter = 0
    guid: int
    highest_guid = 0

    def __init__(self, weight, in_node, out_node):
        self.in_node = in_node
        self.out_node = out_node
        self.enabled = True
        self.weight = weight

    @classmethod
    def new(cls, weight, in_node, out_node, guid=None):
        obj = NNConnection(weight, in_node, out_node)
        obj.innovation_id = cls.global_innovation_counter
        cls.global_innovation_counter += 1
        if guid is None:
            obj.guid = cls.highest_guid
            cls.highest_guid += 1
        else:
            obj.guid = guid
        return obj

    @classmethod
    def regenerate(cls, weight, in_node, out_node, innovation_id, guid, enabled):
        obj = NNConnection(weight, in_node, out_node)
        obj.guid = guid
        obj.enabled = enabled
        obj.innovation_id = innovation_id
        return obj


@dataclass
class BodyPhenotype(AbstractBodyPhenotype):
    body_nodes: list[BNode]
    body_connections: list[BConnection]

    def get(self) -> list:
        return [self.body_nodes, self.body_connections]


@dataclass
class MindPhenotype(AbstractMindPhenotype):
    neural_network_nodes: list[NNNode]
    neural_network_connections: list[NNConnection]

    def reset(self):
        for node in self.neural_network_nodes:
            node.activation_strength = -1

    def get_layers(self) -> list[set]:
        output_layer = set()
        for node in self.neural_network_nodes:
            if node.mode == "output":
                output_layer.add(node)
        next_layer = set()
        for node in output_layer:
            for conn in node.inputs:
                next_layer.add(conn.in_node)
        res = self.recursive_helper(next_layer)
        res.append(output_layer)
        return res

    # TODO: move this to a separate implementation that deals with matrix-representable networks
    #  which defines the following mutations: layer insertion (default everything to everything connections),
    #  add/remove node, add/remove connection
    def get_matrices(self, layers: list[set]):
        matrices = []
        while True:
            current_layer = next(layers)
            print(current_layer.__len__())
            try:
                next_layer = next(layers)
                matrix = np.empty((current_layer.__len__(), next_layer.__len__()))
                i = 0
                for node in current_layer:
                    vals = []
                    for conn in node.outputs:
                        vals.append([conn.weight])
                    np.insert(matrix, i, vals, axis=0)
                    i += 1
                matrices.append(matrix)
            except StopIteration:
                break
        return matrices

    def recursive_helper(self, current_layer):
        next_layer = set()
        for node in current_layer:
            for conn in node.inputs:
                if conn.in_node is not None:
                    next_layer.add(conn.in_node)
        if next_layer.__len__() != 0:
            res = self.recursive_helper(next_layer)
            res.append(current_layer)
            return res
        else:
            return [current_layer]

    def get(self) -> list:
        return [self.neural_network_nodes, self.neural_network_connections]


class CMutator(Mutator):
    mutation_chance: int  # from 0 to 100
    mutation_rolls: int
    current: Optional[Individual]
    max_change: int

    def __init__(self, chance: int, rolls: int, max_change: int):
        self.mutation_chance = chance
        self.mutation_rolls = rolls
        self.current = None
        self.functions = [
            self.new_bconn,
            self.new_bnode,
            self.new_nnnode,
            self.new_nnconn,
            self.weight_shift,
            self.strength_shift,
            self.pos_shift,
            # self.toggle_bconn,
            # self.toggle_nnconn,
            self.remove_nnconn,
            self.remove_bconn,
            self.remove_bnode,
            self.remove_nnnode,
        ]
        self.max_change = max_change

    def new_bconn(self):
        if self.current is None:
            return
        tmp = copy.deepcopy(self.current.genome['body_nodes'])
        pair = list()
        for i in range(2):
            choice = random.choice(tmp)
            tmp.remove(choice)
            pair.append(choice)
        for conn in self.current.genome['body_connections']:
            if conn['in_node'] == pair[0]['guid'] and conn['out_node'] == pair[1]['guid']:
                return
            if conn['in_node'] == pair[1]['guid'] and conn['out_node'] == pair[0]['guid']:
                return
        self.current.genome['body_connections'].append({
            'in_node': pair[0]['guid'],
            'out_node': pair[1]['guid'],
            'weight': random.randint(-1, 1),
            'strength': True,
            'new': True,
            'guid': BConnection.highest_guid,
            'enabled': True
        })
        BConnection.highest_guid += 1

    # tODO: figure out why new bnodes don't seem to get added
    def new_bnode(self):
        if self.current is None:
            return
        conn = random.choice(self.current.genome['body_connections'])
        for nnnode in self.current.genome['nn_nodes']:
            if nnnode['actuator'] == conn['guid']:
                return
        in_node_guid = conn['in_node']
        out_node_guid = conn['out_node']
        self.current.genome['body_connections'].remove(conn)
        in_node_data = None
        out_node_data = None
        for bnode in self.current.genome['body_nodes']:
            if bnode['guid'] == in_node_guid:
                in_node_data = bnode
            if bnode['guid'] == out_node_guid:
                out_node_data = bnode
            if (in_node_data is not None) and (out_node_data is not None):
                break
        if (in_node_data is None) or (out_node_data is None):
            return
        new_bnode = {
            'x': (in_node_data['x'] + out_node_data['x']) / 2,
            'y': (in_node_data['y'] + out_node_data['y']) / 2,
            'guid': BNode.highest_guid,
            'physical_weight': 10
        }
        BNode.highest_guid += 1
        new_conn_1 = {
            'in_node': in_node_guid,
            'out_node': new_bnode['guid'],
            'strength': conn['strength'],
            'enabled': True,
            'new': True,
            'guid': BConnection.highest_guid
        }
        BConnection.highest_guid += 1
        new_conn_2 = {
            'in_node': new_bnode['guid'],
            'out_node': out_node_guid,
            'strength': conn['strength'],
            'enabled': True,
            'new': True,
            'guid': BConnection.highest_guid
        }
        BConnection.highest_guid += 1
        self.current.genome['body_connections'].append(new_conn_1)
        self.current.genome['body_connections'].append(new_conn_2)
        self.current.genome['body_nodes'].append(new_bnode)
        # self.current.genome = self.current.codec.encode(self.current.codec.decode(self.current.genome))
        # print(self.current.genome)
        # self.current.codec.cache = None

    def new_nnnode(self):
        if self.current is None:
            return
        if len(self.current.genome['nn_connections']) < 1:
            return
        conn = random.choice(self.current.genome['nn_connections'])
        in_node_guid = conn['in_node']
        out_node_guid = conn['out_node']
        self.current.genome['nn_connections'].remove(conn)
        in_node_data = None
        out_node_data = None
        for nnnode in self.current.genome['body_nodes']:
            if nnnode['guid'] == in_node_guid:
                in_node_data = nnnode
            if nnnode['guid'] == out_node_guid:
                out_node_data = nnnode
            if (in_node_data is not None) and (out_node_data is not None):
                break
        if (in_node_data is None) or (out_node_data is None):
            return
        new_nnnode = {
            'guid': NNNode.highest_guid,
            'sensor': None,
            'actuator': None,
            'mode': "hidden"
        }
        NNNode.highest_guid += 1
        new_conn_1 = {
            'in_node': in_node_guid,
            'out_node': new_nnnode['guid'],
            'weight': conn['weight'],
            'enabled': True,
            'new': True,
            'guid': NNConnection.highest_guid
        }
        NNConnection.highest_guid += 1

        new_conn_2 = {
            'in_node': new_nnnode['guid'],
            'out_node': out_node_guid,
            'weight': conn['weight'],
            'enabled': True,
            'new': True,
            'guid': NNConnection.highest_guid
        }
        NNConnection.highest_guid += 1

        self.current.genome['nn_connections'].append(new_conn_1)
        self.current.genome['nn_connections'].append(new_conn_2)
        self.current.genome['nn_nodes'].append(new_nnnode)

    def new_nnconn(self):
        if self.current is None:
            return
        tmp = copy.deepcopy(self.current.genome['nn_nodes'])
        pair = list()
        for i in range(2):
            choice = random.choice(tmp)
            tmp.remove(choice)
            pair.append(choice)
        for conn in self.current.genome['nn_connections']:
            if conn['in_node'] == pair[0]['guid'] and conn['out_node'] == pair[1]['guid']:
                return
            if conn['in_node'] == pair[1]['guid'] and conn['out_node'] == pair[0]['guid']:
                return
        self.current.genome['nn_connections'].append({
            'in_node': pair[0]['guid'],
            'out_node': pair[1]['guid'],
            'weight': random.randint(-1, 1),
            'enabled': True,
            'new': True,
            'guid': NNConnection.highest_guid
        })
        NNConnection.highest_guid += 1

    def weight_shift(self):
        if len(self.current.genome['nn_connections']) < 1:
            return
        conn = random.choice(self.current.genome['nn_connections'])
        sign = random.choice([1, -1])
        magnitude = random.randint(1, self.max_change)
        weight = conn['weight']
        conn['weight'] = weight + (sign * weight * (magnitude / 100))

    def strength_shift(self):
        if len(self.current.genome['body_connections']) < 1:
            return
        conn = random.choice(self.current.genome['body_connections'])
        sign = random.choice([1, -1])
        magnitude = random.randint(1, self.max_change)
        strength = conn['strength']
        conn['strength'] = strength + (sign * strength * (magnitude / 100))

    def pos_shift(self):
        if len(self.current.genome['body_nodes']) < 1:
            return
        bnode = random.choice(self.current.genome['body_nodes'])
        sign = random.choice([1, -1])
        sign2 = random.choice([1, -1])
        magnitude = random.randint(1, self.max_change)
        magnitude2 = random.randint(1, self.max_change)
        new_x = bnode['x'] + (sign * bnode['x'] * (magnitude / 100))
        new_y = bnode['y'] + (sign2 * bnode['y'] * (magnitude2 / 100))
        bnode['x'] = new_x
        bnode['y'] = new_y

    def phys_weight_shift(self):
        if len(self.current.genome['body_nodes']) < 1:
            return
        choice = random.choice(self.current.genome['body_nodes'])
        sign = random.choice([-1, 1])
        choice['physical_weight'] += sign * choice['physical_weight'] * (self.max_change / 100)

    def toggle_nnconn(self):
        if len(self.current.genome['nn_connections']) <= 1:
            return
        conn = random.choice(self.current.genome['nn_connections'])
        conn['enabled'] = not conn['enabled']

    def remove_nnconn(self):
        if len(self.current.genome['nn_connections']) <= 1:
            return
        conn = random.choice(self.current.genome['nn_connections'])
        self.current.genome['nn_connections'].remove(conn)

    def remove_bconn(self):
        if len(self.current.genome['body_connections']) <= 1:
            return
        conn = random.choice(self.current.genome['body_connections'])
        for output in self.current.genome['nn_nodes']:
            if output['actuator'] == conn['guid']:
                return
        self.current.genome['body_connections'].remove(conn)

    def toggle_bconn(self):
        if len(self.current.genome['body_connections']) <= 1:
            return
        conn = random.choice(self.current.genome['body_connections'])
        conn['enabled'] = not conn['enabled']

    def remove_bnode(self):
        node = random.choice(self.current.genome['body_nodes'])
        conns = list()
        for conn in self.current.genome['body_connections']:
            if conn['in_node'] == node['guid'] or conn['out_node'] == node['guid']:
                conns.append(conn)
        for conn in conns:
            for output in self.current.genome['nn_nodes']:
                if output['actuator'] == conn['guid']:
                    return
        self.current.genome['body_nodes'].remove(node)
        for conn in conns:
            self.current.genome['body_connections'].remove(conn)

    def remove_nnnode(self):
        node = random.choice(self.current.genome['nn_nodes'])
        if node['mode'] != "hidden":
            return
        conns = list()
        for conn in self.current.genome['nn_connections']:
            if conn['in_node'] == node['guid'] or conn['out_node'] == node['guid']:
                conns.append(conn)
        self.current.genome['nn_nodes'].remove(node)
        for conn in conns:
            self.current.genome['nn_connections'].remove(conn)

    def add_sensor(self):
        grp1 = ["SensorClock20", "SensorClock100",
                "InputZeroOne", "InputPlusMinus", "SensorSignClock20",
                "SensorMeanX", "SensorMeanY"]
        grp2 = ["SensorNodeX", "SensorNodeY"]
        roll = random.randint(0, 1)
        if roll == 1:
            cls = random.choice(grp2)
            chosen = random.choice(self.current.genome['body_nodes'])
            self.current.genome['nn_nodes'].append({
                'mode': "input",
                'actuator': None,
                'sensor': (cls, chosen['guid']),
                'guid': NNNode.highest_guid
            })
            NNNode.highest_guid += 1
        elif roll == 0:
            cls = random.choice(grp1)
            self.current.genome['nn_nodes'].append({
                'mode': "input",
                'actuator': None,
                'sensor': (cls, None),
                'guid': NNNode.highest_guid
            })
            NNNode.highest_guid += 1

    def add_actuator(self):
        excluded = list()
        guids = list()
        for node in self.current.genome['nn_nodes']:
            if node["output"]:
                excluded.append(node['actuator'])
                guids.append(node['guid'])
        retries = 0
        chosen = None
        while True:
            chosen = self.current.genome['body_connections']
            if chosen['guid'] in excluded:
                retries += 1
                if retries > 4:
                    return
            else:
                break

        self.current.genome['nn_nodes'].append({
            'mode': "output",
            'actuator': chosen['guid'],
            'sensor': None,
            'guid': NNNode.highest_guid
        })
        saved_guid = NNNode.highest_guid
        NNNode.highest_guid += 1

        to_connect = list()

        for conn in self.current.genome['nn_connections']:
            if conn['out_node'] in guids:
                to_connect.append(conn['in_node'])
        for guid in to_connect:
            self.current.genome['nn_connections'].append({
                'in_node': guid,
                'out_node': saved_guid,
                'weight': random.randint(-1, 1),
                'enabled': True,
                'new': True,
                'guid': NNConnection.highest_guid
            })
            NNConnection.highest_guid += 1

    def remove_actuator(self):
        chosen = None
        for node in self.current.genome['nn_nodes']:
            if node['mode'] == "output":
                chosen = node
                break
        if chosen is None:
            return
        for conn in self.current.genome['nn_connections']:
            if conn['out_node'] == chosen['guid']:
                self.current.genome['nn_connections'].remove(conn)
        self.current.genome['nn_nodes'].remove(chosen)

    def mutate(self, population_members: list[Individual]) -> list[Individual]:
        print('mutator')
        # return population_members
        for individual in population_members:
            self.current = individual
            self.current.codec.cache = None
            # TODO: enable mutation chance
            for i in range(0, self.mutation_rolls):
                if random.randint(0, 100) < self.mutation_chance:
                    verb = random.choice(self.functions)
                    verb()
                    self.current.codec.cache = None
                    self.current.encode(self.current.decode())

        return population_members


class CSelector(Selector):
    compatibility_threshold: float
    culling_threshold: float
    coefficient_w: float
    coefficient_e: float
    coefficient_d: float

    def __init__(self, compatibility_threshold: float, culling_threshold: float, coefficient_w: float, coefficient_e: float,
                 coefficient_d: float, starting_pop_number: int):
        self.coefficient_e = coefficient_e
        self.coefficient_d = coefficient_d
        self.compatibility_threshold = compatibility_threshold
        self.coefficient_w = coefficient_w
        self.last_species_alive_count = 1
        self.culling_threshold = culling_threshold
        self.pop_size = starting_pop_number

    def select(self, population_members: list[Individual]) -> list[Individual]:
        pop_size = len(population_members)
        print('pop_size: ', pop_size)
        print('compt_threshold: ', self.compatibility_threshold)
        # Culling genomes causing simulation errors
        for member in population_members:
            if member.sim_data.cull:
                population_members.remove(member)
        
        self.adjust_fitness(population_members)
        self.speciate(population_members)
        mean_adjusted_fitness = 0
        species = []
        children = []
        species_diagnostic = dict()
        for i in range(0, Individual.species_counter):
            species.append([])
        for individual in population_members:
            mean_adjusted_fitness += individual.fitness()
            species[individual.species].append(individual)
            species_diagnostic[str(individual.species)] = 1
        mean_adjusted_fitness = mean_adjusted_fitness / len(population_members)
        average_species_fitness = []

        if len(species_diagnostic) < self.last_species_alive_count:
            print('Loss of diversity detected, adjusting species compatibility threshold')
            self.compatibility_threshold *= 1.001 - len(species_diagnostic) / self.last_species_alive_count
            print('New threshold: ', self.compatibility_threshold)

        print('no. species: ', len(species_diagnostic))
        self.last_species_alive_count = len(species_diagnostic)
        for group in species:
            if len(group) == 0:
                continue
            if len(group) == 1 and group[0].newly_speciated:
                for i in range(0, 3):
                    group.append(copy.deepcopy(group[0]))
            elif len(group) == 1:
                continue
            avg = 0
            for individual in group:
                avg += individual.fitness()
            avg = avg / len(group)
            average_species_fitness.append(avg)
            new_size = math.ceil(avg / mean_adjusted_fitness * len(group))

            # culling outliers which fell below the threshold
            old_len = len(group)
            for member in group:
                if member.fitness() < self.culling_threshold * avg:
                    group.remove(member)
            if len(group) == 0:
                continue

            # recouping culling losses
            try:
                if len(group) < old_len:
                    fitness_scores = [m.fitness_tester.test(m) for m in group]
                    base = random.choices(group, fitness_scores, k=1)[0]
                    group.append(copy.deepcopy(base))
                group.sort(key=lambda x: x.fitness(), reverse=True)
            except ValueError:
                continue

            # adjusting species size
            if new_size > len(group):
                for i in range(0, (new_size - len(group))):
                    fitness_scores = [m.fitness_tester.test(m) for m in group]
                    base = random.choices(group, fitness_scores, k=1)[0]
                    children.append(copy.deepcopy(base))
                for individual in group:
                    children.append(individual)
            else:
                for i in range(0, new_size):
                    children.append(group[i])

        # recouping losses to culling and species extinctions
        if len(children) < self.pop_size:
            fitness_scores = [m.fitness_tester.test(m) for m in population_members]
            for i in range(0, (self.pop_size-len(children))):
                base = random.choices(population_members, fitness_scores, k=1)[0]
                children.append(copy.deepcopy(base))
        if len(children) > self.pop_size:
            children.sort(reverse=True, key=lambda x: x.fitness())
            for i in range(0, (len(children)-self.pop_size)):
                children.pop()
        return children

    def speciate(self, population_members: list[Individual]):
        for individual in population_members:
            deltas = dict()
            for individual2 in population_members:
                if individual2 == individual:
                    continue
                delta = self.get_delta(individual, individual2)
                deltas[delta] = individual2
            deltas = sorted(deltas.items())
            if deltas[0][0] < self.compatibility_threshold:
                closest = deltas[0][1]
                individual.species = closest.species
                continue
            individual.species = Individual.species_counter
            Individual.species_counter += 1
            individual.newly_speciated = True

    def get_delta(self, ind1: Individual, ind2: Individual) -> float:
        # N = max(len(ind1.genome['body_nodes']) + len(ind1.genome['nn_nodes']), len(ind2.genome['body_nodes']) +
        #         len(ind2.genome['nn_nodes']))
        N = max(len(ind1.genome['body_connections']) + len(ind1.genome['nn_connections']),
                len(ind2.genome['body_connections']) + len(ind2.genome['nn_connections']))
        excess = 0
        disjoint = 0
        weight_diffs = []
        range_bconns = 0
        range_nnconns = 0
        lrange_bconns = 9999999
        lrange_nnconns = 9999999

        if len(ind1.genome['nn_connections']) > len(ind2.genome['nn_connections']):
            bigger_nn = ind1
            smaller_nn = ind2
        else:
            bigger_nn = ind2
            smaller_nn = ind1
        if len(ind1.genome['body_connections']) > len(ind2.genome['body_connections']):
            bigger_bod = ind1
            smaller_bod = ind2
        else:
            bigger_bod = ind2
            smaller_bod = ind1

        for gene in ind1.genome['body_connections']:
            if gene['innovation_id'] > range_bconns:
                range_bconns = gene['innovation_id']
            if gene['innovation_id'] < lrange_bconns:
                lrange_bconns = gene['innovation_id']
        for gene in ind1.genome['nn_connections']:
            if gene['innovation_id'] > range_nnconns:
                range_nnconns = gene['innovation_id']
            if gene['innovation_id'] < lrange_nnconns:
                lrange_nnconns = gene['innovation_id']

        for gene in bigger_bod.genome['body_connections']:
            if gene['innovation_id'] < lrange_bconns or gene['innovation_id'] > range_bconns:
                excess += 1
                continue
            flag = True
            for gene2 in smaller_bod.genome['body_connections']:
                if gene['innovation_id'] == gene2['innovation_id']:
                    weight_diffs.append(abs(gene['strength'] - gene2['strength']))
                    flag = False
                    break
            if flag:
                disjoint += 1

        for gene in bigger_nn.genome['nn_connections']:
            if gene['innovation_id'] < lrange_nnconns or gene['innovation_id'] > range_nnconns:
                excess += 1
                continue
            flag = True
            for gene2 in smaller_nn.genome['nn_connections']:
                if gene['innovation_id'] == gene2['innovation_id']:
                    weight_diffs.append(abs(gene['weight'] - gene2['weight']))
                    flag = False
                    break
            if flag:
                disjoint += 1

        avg_weights_diff = sum(weight_diffs) / len(weight_diffs)
        return ((self.coefficient_e * excess) / N) \
            + ((self.coefficient_d * disjoint) / N) \
            + self.coefficient_w * avg_weights_diff

    def adjust_fitness(self, population_members: list[Individual]):
        if population_members[0].generation == 1:
            return
        species = defaultdict(list)
        for specimen in population_members:
            species[specimen.species].append(specimen)
        grouped = species.values()
        for group in grouped:
            if len(group) == 0:
                continue
            for specimen in group:
                specimen.adjuster = len(group)


class CCrossover(Crossover):
    def crossover(self, population_members: list[Individual]) -> list[Individual]:
        # return population_members
        # problem: crossed-over genes will have to have their nodes and connections rewired - nontrivial problem
        # can try to patch connections to genes matching the innovation id of the previous connection endpoints???
        children = list()
        species = list()
        mating_pairs = list()
        for i in range(0, Individual.species_counter):
            species.append(list())
        for individual in population_members:
            species[individual.species].append(individual)
        for group in species:
            while group.__len__() > 1:
                specimen1 = random.choice(group)
                group.remove(specimen1)
                specimen2 = random.choice(group)
                group.remove(specimen2)
                pair = (specimen2, specimen2)
                mating_pairs.append(pair)
            if group.__len__() == 1:
                if len(mating_pairs) <= 1:
                    continue
                specimen1 = group.pop()
                specimen2 = copy.deepcopy(random.choice(mating_pairs)[random.choice([0, 1])])
                pair = (specimen1, specimen2, 'LIMITED')
                mating_pairs.append(pair)
        for pair in mating_pairs:
            innovations1 = dict()
            innovations1nn = dict()
            innovations2 = dict()
            innovations2nn = dict()
            offspring_genomes = list()
            for i in range(0, 2):
                offspring_genomes.append({
                    'body_nodes': list(),
                    'body_connections': list(),
                    'nn_nodes': list(),
                    'nn_connections': list(),
                    'species': pair[0].species,
                    'generation': pair[0].generation
                })

            for gene in pair[0].genome['body_connections']:
                innovations1[gene['innovation_id']] = gene
            for gene in pair[0].genome['nn_connections']:
                innovations1nn[gene['innovation_id']] = gene

            for gene in pair[1].genome['body_connections']:
                innovations2[gene['innovation_id']] = gene
            for gene in pair[1].genome['nn_connections']:
                innovations2nn[gene['innovation_id']] = gene

            for i in range(0, 1):
                for innov, gene in innovations1.items():
                    if innov in innovations2.keys():
                        choice = random.choice((gene, innovations2[innov]))
                        offspring_genomes[i]['body_connections'].append(choice)
                    else:
                        offspring_genomes[i]['body_connections'].append(gene)
                for innov, gene in innovations1nn.items():
                    if innov in innovations2nn.keys():
                        choice = random.choice((gene, innovations2nn[innov]))
                        offspring_genomes[i]['nn_connections'].append(choice)
                    else:
                        offspring_genomes[i]['nn_connections'].append(gene)
                for gene in pair[0].genome['body_nodes']:
                    offspring_genomes[i]['body_nodes'].append(gene)
                for gene in pair[0].genome['nn_nodes']:
                    offspring_genomes[i]['nn_nodes'].append(gene)

            for i in range(1, 2):
                for innov, gene in innovations2.items():
                    if innov in innovations1.keys():
                        choice = random.choice((gene, innovations1[innov]))
                        offspring_genomes[i]['body_connections'].append(choice)
                    else:
                        offspring_genomes[i]['body_connections'].append(gene)
                for innov, gene in innovations2nn.items():
                    if innov in innovations1nn.keys():
                        choice = random.choice((gene, innovations1nn[innov]))
                        offspring_genomes[i]['nn_connections'].append(choice)
                    else:
                        offspring_genomes[i]['nn_connections'].append(gene)
                for gene in pair[1].genome['body_nodes']:
                    offspring_genomes[i]['body_nodes'].append(gene)
                for gene in pair[1].genome['nn_nodes']:
                    offspring_genomes[i]['nn_nodes'].append(gene)

            if len(pair) == 3:
                choice = random.choice(offspring_genomes)
                ind = Individual(CEncoderDecoder(), choice)
                ind.species = choice['species']
                ind.generation = choice['generation'] + 1
                continue
            for genome in offspring_genomes:
                ind = Individual(CEncoderDecoder(), genome)
                ind.species = genome['species']
                ind.generation = genome['generation'] + 1
                children.append(ind)

        return children

        # for individual in population_members:
        #     phenotype = individual.decode()
        #     partner = random.choice(population_members)
        #     while partner.species != individual.species:
        #         partner = random.choice(population_members)
        #     phenotype2 = partner.decode()
        #     child = Individual(individual.codec, [])
        #     child_phenotype = Phenotype(BodyPhenotype([], []), MindPhenotype([], []))
        #     if individual.fitness() > partner.fitness():
        #         for gene in phenotype.mind.get()[1]:
        #             matching_found = False
        #             for gene2 in phenotype2.mind.get()[1]:
        #                 if gene.innovation_id == gene2.innovation_id:
        #                     child_phenotype.mind.get()[1].append(random.choice([gene, gene2]))
        #                     matching_found = True
        #                     break
        #             if not matching_found:
        #                 child_phenotype.mind.get()[1].append(gene)
        #         for gene in phenotype.body.get()[1]:
        #             matching_found = False
        #             for gene2 in phenotype2.body.get()[1]:
        #                 if gene.innovation_id == gene2.innovation_id:
        #                     child_phenotype.body.get()[1].append(random.choice([gene, gene2]))
        #                     matching_found = True
        #                     break
        #             if not matching_found:
        #                 child_phenotype.body.get()[1].append(gene)
        #     else:
        #         for gene in phenotype2.mind.get()[1]:
        #             matching_found = False
        #             for gene2 in phenotype.mind.get()[1]:
        #                 if gene.innovation_id == gene2.innovation_id:
        #                     child_phenotype.mind.get()[1].append(random.choice([gene, gene2]))
        #                     matching_found = True
        #                     break
        #             if not matching_found:
        #                 child_phenotype.mind.get()[1].append(gene)
        #         for gene in phenotype2.body.get()[1]:
        #             matching_found = False
        #             for gene2 in phenotype.body.get()[1]:
        #                 if gene.innovation_id == gene2.innovation_id:
        #                     tmp = [gene, gene2]
        #                     choice = random.choice([gene, gene2])
        #                     tmp.remove(choice)
        #                     clone = copy.deepcopy(choice)
        #                     clone.inputs = copy.deepcopy(tmp[0].inputs)
        #                     clone.outputs = copy.deepcopy(tmp[0].outputs)
        #                     child_phenotype.body.get()[1].append(clone)
        #                     matching_found = True
        #                     break
        #                 else:
        #                     child_phenotype.body.get()[1].append(gene)
        #             if not matching_found:
        #                 child_phenotype.body.get()[1].append(gene)
        #     child.encode(child_phenotype)
        #     child.species = individual.species
        #     children.append(child)


class CGenomeInitializer(GenomeInitializer):
    prototype: dict

    def create(self):
        bnode1 = BNode.new(100, 100)
        bnode2 = BNode.new(130, 270)
        bnode3 = BNode.new(200, 200)
        bnode4 = BNode.new(100, 200)
        # bnode4 = BNode(130, 205)
        bconn1 = BConnection.new(bnode1, bnode2, 0.5)
        bconn2 = BConnection.new(bnode2, bnode3, 0.5)
        bconn3 = BConnection.new(bnode3, bnode1, 0.5)
        bconn4 = BConnection.new(bnode1, bnode4, 0.5)
        bconn5 = BConnection.new(bnode2, bnode4, 0.5)
        bconn6 = BConnection.new(bnode3, bnode4, 0.5)
        # bconn4 = BConnection(bnode4, bnode1, 0.5)
        # bconn5 = BConnection(bnode2, bnode4, 0.5)
        # bconn6 = BConnection(bnode3, bnode4, 0.5)
        bnode1.inputs.append(bconn3)
        bnode2.inputs.append(bconn1)
        bnode3.inputs.append(bconn2)
        bnode1.outputs.append(bconn1)
        bnode1.outputs.append(bconn4)
        bnode2.outputs.append(bconn2)
        bnode2.outputs.append(bconn5)
        bnode3.outputs.append(bconn3)
        bnode3.outputs.append(bconn6)
        bnode4.inputs.append(bconn4)
        bnode4.inputs.append(bconn5)
        bnode4.inputs.append(bconn6)
        # bnode4.outputs.append(bconn4)
        # bnode1.inputs.append(bconn4)
        # bnode2.outputs.append(bconn5)
        # bnode3.outputs.append(bconn6)
        # possible circularity bug?
        # input layer
        inputs = []
        input_types = [
            # SensorNodeX(node=bnode1, sim_data=None, label=SensorNodeX.label),
            # SensorNodeY(node=bnode1, sim_data=None, label=SensorNodeY.label),
            SensorSignClock20(node=None, sim_data=None, label=''),
            SensorMeanX(node=None, sim_data=None, label=SensorMeanX.label),
            SensorClock20(node=None, sim_data=None, label='')
            # SensorTime(node=None, sim_data=None, label=SensorTime.label),
        ]
        for sensor in input_types:
            # node = NNNode("input", inputs=[], outputs=[], sensor=sensor)
            node = NNNode.new("input", sensor)
            inputs.append(node)

        hidden = []
        in_conns = []

        for i in range(0, 3):
            local_conns = []
            for parent in inputs:
                # conn = NNConnection(weight=1, in_node=parent, out_node=None)
                conn = NNConnection.new(1, parent, None)
                in_conns.append(conn)
                parent.outputs.append(conn)
                local_conns.append(conn)
            # node = NNNode("hidden", inputs=in_conns, outputs=[])
            node = NNNode.new("hidden")
            node.inputs = in_conns
            for conn in local_conns:
                conn.out_node = node
            hidden.append(node)

        # output = NNNode("output", inputs=[], outputs=[], actuator=bconn1)
        output = NNNode.new("output", actuator=bconn1)
        output2 = NNNode.new("output", actuator=bconn3)
        out_conns = []
        for node in hidden:
            # conn = NNConnection(weight=1, in_node=node, out_node=output)
            conn = NNConnection.new(1, node, output)
            conn2 = NNConnection.new(1, node, output2)
            out_conns.append(conn)
            out_conns.append(conn2)
            node.outputs.append(conn)
            node.outputs.append(conn2)
            output.inputs.append(conn)
            output.inputs.append(conn2)

        codec = CEncoderDecoder()
        body_nodes = [bnode1, bnode2, bnode3, bnode4]
        body_connections = [bconn1, bconn2, bconn3, bconn4, bconn5, bconn6]
        nn_nodes = inputs + hidden + [output, output2]
        nn_connections = in_conns + out_conns
        genome = codec.encode(Phenotype(BodyPhenotype(body_nodes, body_connections),
                                        MindPhenotype(nn_nodes, nn_connections)))
        # result = [
        #     [
        #         [bnode1, bnode2, bnode3,
        #          # bnode4
        #          ],  # body nodes
        #         [bconn1, bconn2, bconn3,
        #          # bconn4, bconn5, bconn6
        #          ]   # body connections
        #     ],
        #     [
        #         inputs + hidden + [output],  # neural network nodes
        #         in_conns + out_conns   # neural network connections
        #     ]
        # ]
        self.prototype = genome


class PreloaderGenomeInitializer(GenomeInitializer):
    prototype: dict

    def __init__(self, genome):
        self.prototype = genome

    def create(self):
        return self.prototype

from __future__ import annotations

import copy
import math
import os
import random
import statistics

from pymunk import Vec2d

from concrete_evolution import BNode
from pymunk.constraints import DampedSpring
from interfaces import evolution as ne
from interfaces.simulation import Simulation, SimParams, SimData
import pymunk
import pygame

COLLTYPE_BALL = 2

def flipy(y):
    """Small hack to convert chipmunk physics to pygame coordinates"""
    # Code fragment taken from the official pymunk documentation
    return -y + 600

def check_looping(network) -> bool:
    pass


class CSimulation(Simulation):
    def __init__(self, sim_params: SimParams):
        super().__init__(sim_params)

    def run_sim(self, specimen: ne.Individual, pos):
        os.environ['SDL_VIDEO_WINDOW_POS'] = '%d,%d' % (pos[0], pos[1])
        self.sim_data = SimData(0, 0, 0, False, 0)
        self.join_sim_data(specimen)

        space = pymunk.Space()
        space.gravity = self.sim_params.gravity
        line_point1 = Vec2d(-100000, 10)
        line_point2 = Vec2d(100000, 10)
        ground = pymunk.Segment(
            space.static_body, line_point1, line_point2, 0.0
        )
        ground.friction = 0.99
        space.add(ground)

        for gene in specimen.decode().body.get()[0]:
            node = pymunk.Body(10, 100)
            node.position = Vec2d(gene.x, gene.y)
            node_shape = pymunk.Circle(node, gene.physical_weight, (0, 0))
            node_shape.friction = 0.5
            node_shape.collision_type = COLLTYPE_BALL
            space.add(node, node_shape)
            gene.physical_node = node
            gene.physical_shape = node_shape

        for node in specimen.decode().body.get()[0]:
            for connection in node.outputs:
                if not connection.enabled:
                    continue
                distance = math.sqrt((node.x - connection.out_node.x) ** 2 + (node.y - connection.out_node.y) ** 2)
                spring = DampedSpring(node.physical_node, connection.out_node.physical_node,
                                      stiffness=(1-connection.strength)*5000, anchor_a=(0, 0), anchor_b=(0, 0),
                                      damping=(1 - connection.strength)*1000, rest_length=distance)
                connection.physical_dist = distance
                connection.physical_spring = spring
                node.physical_connections.append({"spr": spring, "conn": connection})
                space.add(spring)
        dt = 1.0 / 60.0
        print_options = pymunk.SpaceDebugDrawOptions()
        pygame.init()
        # self.font = pygame.font.Font("assets/fonts/arial.ttf", 15)
        screen = pygame.display.set_mode((600, 600))
        clock = pygame.time.Clock()

        for x in range(self.sim_params.time):
            self.sim_data.time = x
            grounda = int(line_point1.x), int(flipy(line_point1.y))
            groundb = int(line_point2.x), int(flipy(line_point2.y))
            space.step(dt)
            screen.fill(pygame.Color("white"))
            pygame.draw.line(screen, pygame.Color("red"), grounda, groundb)

            for gene in specimen.decode().body.get()[0]:
                # gene.physical_shape.angle = 0
                # gene.physical_node.angle = 0
                gene.physical_node.angular_velocity = 0
                # print(gene.physical_node.angle)
                # min(255, abs(int(gene.physical_node.angle) * 20))
                r = gene.physical_shape.radius
                v = gene.physical_node.position
                rot = gene.physical_node.rotation_vector
                p = int(v.x), int(flipy(v.y))
                pygame.draw.circle(screen, pygame.Color("blue"), p, int(r))
                pygame.draw.line(screen, pygame.Color('red'), p, (p[0]+(rot.x*10), p[1]+(rot.y*10)))
                for spring in gene.physical_connections:
                    v1 = spring["spr"].a.position
                    v2 = spring["spr"].b.position
                    p1 = int(v1.x), int(flipy(v1.y))
                    p2 = int(v2.x), int(flipy(v2.y))
                    pygame.draw.line(screen, pygame.Color(spring["conn"].color), p1, p2)
                    spring["conn"].color = "black"

            cord1 = 550
            cord2 = flipy(550)
            next_layer = list()
            num_of_inputs = 0

            self.update_sim_data(specimen)
            try:
                self.update_lengths(specimen)
            except RecursionError as re:
                print("Cycle detected, culling specimen")
                specimen.sim_data = self.sim_data
                specimen.sim_data.mean_x = -20
                specimen.sim_data.cull = True
                return specimen
            # self.draw_neural_net(cord1, cord2, next_layer, num_of_inputs, screen, specimen)
            specimen.decode().mind.reset()

            pygame.display.flip()
            clock.tick(120)
            pygame.display.set_caption("fps: " + str(clock.get_fps()))
            if x == 0 and self.sim_params.save_picture == 1:
                if not os.path.isdir(self.sim_params.save_directory+"/images"):
                    os.mkdir(self.sim_params.save_directory + "/images/")
                if not os.path.isdir(self.sim_params.save_directory+"/images/"+str(specimen.generation)):
                    os.mkdir(self.sim_params.save_directory+"/images/"+str(specimen.generation))
                pygame.image.save(screen, self.sim_params.save_directory+"/images/"+str(specimen.generation)+"/"
                                  + str(specimen.guid)+".png")
        self.sim_data.bounded_area = self.get_bounded_area(specimen)
        if self.sim_data.bounded_area == 0:
            self.sim_data.cull = True
        specimen.sim_data = self.sim_data
        return specimen

# TODO: make this draw non-layer graphs, figure out rough position by number of connections to closest input and output
    def draw_neural_net(self, cord1, cord2, next_layer, num_of_inputs, screen, specimen):
        for node in specimen.decode().mind.get()[0]:
            if node.mode == "input":
                pygame.draw.circle(screen, pygame.Color("blue"), (cord1, cord2), 10)
                display_val = "%.2f" % round(node.activation_strength, 2)
                screen.blit(self.font.render(str(display_val), True, (0, 0, 0)), (cord1, cord2))
                label = node.sensor.label
                screen.blit(self.font.render(str(label), True, (0, 0, 0)), (cord1, cord2-30))
                node.pos = (cord1, cord2)
                cord1 -= 60
                next_layer += node.outputs
                num_of_inputs += 1
        self.recursive_draw(next_layer, cord2, num_of_inputs, screen)

    def update_lengths(self, specimen):
        for node in specimen.decode().mind.get()[0]:
            if node.mode == "output":
                val = node.activate()
                node.actuator.physical_spring.rest_length = node.actuator.physical_dist * val

    def recursive_draw(self, current_layer, cord2, prevnum, screen):
        cord2 += 60
        nodes = set()
        next_layer = list()
        for conn in current_layer:
            nodes.add(conn.out_node)
        number = nodes.__len__()
        span = (number - 1) * 60
        prevspan = (prevnum - 1) * 60
        diff = span - prevspan
        offset = 0
        if diff != 0:
            offset = diff / 2
        cord1 = 550 + offset
        for node in nodes:
            pygame.draw.circle(screen, pygame.Color("blue"), (cord1, cord2), 10)
            display_val = "%.2f" % round(node.activation_strength, 2)
            screen.blit(self.font.render(str(display_val), True, (0, 0, 0)), (cord1, cord2))
            node.pos = (cord1, cord2)
            cord1 -= 60
            if node.outputs:
                next_layer += node.outputs
        for conn in current_layer:
            pygame.draw.line(screen, pygame.Color("black"), conn.in_node.pos, conn.out_node.pos)
        if next_layer:
            self.recursive_draw(next_layer, cord2, number, screen)

    def update_sim_data(self, specimen):
        nodes = specimen.decode().body.get()[0]
        xs = []
        ys = []
        for node in nodes:
            xs.append(node.physical_node.position.x)
            ys.append(node.physical_node.position.y)
        self.sim_data.mean_x = statistics.mean(xs)
        self.sim_data.mean_y = statistics.mean(ys)

    def join_sim_data(self, specimen):
        for node in specimen.decode().mind.get()[0]:
            if node.mode == "input":
                node.sensor.sim_data = self.sim_data

    def get_bounded_area(self, specimen):
        lowest_x = specimen.decode().body.get()[0][0].physical_node.position.x
        highest_x = specimen.decode().body.get()[0][0].physical_node.position.x
        lowest_y = specimen.decode().body.get()[0][0].physical_node.position.y
        highest_y = specimen.decode().body.get()[0][0].physical_node.position.y
        for node in specimen.decode().body.get()[0]:
            if node.physical_node.position.x > highest_x:
                highest_x = node.physical_node.position.x
            elif node.physical_node.position.x < lowest_x:
                lowest_x = node.physical_node.position.x
            if node.physical_node.position.y > highest_y:
                highest_y = node.physical_node.position.y
            elif node.physical_node.position.y < lowest_y:
                lowest_y = node.physical_node.position.y
        len_1 = highest_y - lowest_y
        len_2 = highest_x - lowest_x
        return len_1 * len_2


class CSimulationSilent(CSimulation):
    def run_sim(self, specimen: ne.Individual, pos):
        os.environ['SDL_VIDEO_WINDOW_POS'] = '%d,%d' % (pos[0], pos[1])
        self.sim_data = SimData(0, 0, 0, False, 0)
        self.join_sim_data(specimen)

        space = pymunk.Space()
        space.gravity = self.sim_params.gravity
        line_point1 = Vec2d(-1000, 10)
        line_point2 = Vec2d(1000, 10)
        ground = pymunk.Segment(
            space.static_body, line_point1, line_point2, 0.0
        )
        ground.friction = 0.99
        space.add(ground)

        for gene in specimen.decode().body.get()[0]:
            node = pymunk.Body(10, 100)
            node.position = Vec2d(gene.x, gene.y)
            node_shape = pymunk.Circle(node, gene.physical_weight, (0, 0))
            node_shape.friction = 0.5
            node_shape.collision_type = COLLTYPE_BALL
            space.add(node, node_shape)
            gene.physical_node = node
            gene.physical_shape = node_shape

        for node in specimen.decode().body.get()[0]:
            for connection in node.outputs:
                if not connection.enabled:
                    continue
                distance = math.sqrt((node.x - connection.out_node.x) ** 2 + (node.y - connection.out_node.y) ** 2)
                spring = DampedSpring(node.physical_node, connection.out_node.physical_node,
                                      stiffness=(1-connection.strength)*5000, anchor_a=(0, 0), anchor_b=(0, 0),
                                      damping=(1 - connection.strength)*1000, rest_length=distance)
                connection.physical_dist = distance
                connection.physical_spring = spring
                node.physical_connections.append({"spr": spring, "conn": connection})
                space.add(spring)
        dt = 1.0 / 60.0
        print_options = pymunk.SpaceDebugDrawOptions()
        # pygame.init()
        # self.font = pygame.font.Font("assets/fonts/arial.ttf", 15)
        # screen = pygame.display.set_mode((600, 600))
        clock = pygame.time.Clock()

        for x in range(self.sim_params.time):
            for gene in specimen.decode().body.get()[0]:
                gene.physical_node.angular_velocity = 0
            self.sim_data.time = x
            space.step(dt)

            self.update_sim_data(specimen)
            try:
                self.update_lengths(specimen)
            except RecursionError as re:
                print("Cycle detected, culling specimen")
                specimen.sim_data = self.sim_data
                specimen.sim_data.mean_x = -20
                specimen.sim_data.cull = True
                return specimen
            specimen.decode().mind.reset()

            clock.tick(14400)
        self.sim_data.bounded_area = self.get_bounded_area(specimen)
        if self.sim_data.bounded_area == 0:
            self.sim_data.cull = True
        specimen.sim_data = self.sim_data
        return specimen

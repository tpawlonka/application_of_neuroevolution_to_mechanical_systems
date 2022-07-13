from abc import ABC, abstractmethod
from asyncio import Future
from dataclasses import dataclass


@dataclass
class SimParams:
    gravity: tuple[float, int]
    time: int
    save_picture: int
    save_directory: str


@dataclass
class SimData:
    time: int
    mean_x: float
    mean_y: float
    cull: bool
    bounded_area: float
    middle = 132.5

# @dataclass
# class Result(ABC):
#     sim_record: list
#     body_plan: ne.AbstractBodyPhenotype


class Simulation(ABC):
    sim_params: SimParams

    @abstractmethod
    def __init__(self, sim_params: SimParams):
        self.sim_params = sim_params

    @abstractmethod
    def run_sim(self, specimen, pos):
        pass
        # return fitness.measure(Result(sim_record, body_plan))
        # serialize and save result to a file

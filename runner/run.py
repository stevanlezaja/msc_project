from typing import Callable

from fibers import Fiber
from raman_amplifier import RamanAmplifier


class Runner:
    def __init__(self):
        self.fiber: Fiber = None
        self.raman_amplifier: RamanAmplifier = None
        self.experiment: Callable = None

    def run(self):
        pass

    def set_raman_amplifier(self):
        pass

    def set_signal(self):
        pass

    def set_fiber(self):
        pass

    def set_experiment(self):
        pass
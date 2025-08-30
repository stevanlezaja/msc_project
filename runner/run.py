from typing import Callable
from time import sleep, time

from fibers import Fiber
from raman_amplifier import RamanAmplifier


class Runner:
    def __init__(self):
        self.fiber: Fiber = None
        self.raman_amplifier: RamanAmplifier = None
        self.experiment: Callable = None
        self.running = True
        self.command_buffer: list[Callable] = []
        self.start_time = time()

    def run(self):
        while self.running:
            print(f"Running for {time() - self.start_time}")
            sleep(0.1)

    def set_raman_amplifier(self):
        pass

    def set_signal(self):
        pass

    def set_fiber(self):
        pass

    def set_experiment(self):
        pass
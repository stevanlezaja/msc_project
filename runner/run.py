from typing import Callable
from time import sleep, time

from custom_types import Power, Length
import custom_logging as clog

from fibers import Fiber
from signals import Signal
from raman_amplifier import RamanAmplifier
from experiment.experiment import RamanSystem


log = clog.get_logger("Runner")


class Runner:
    def __init__(self):
        self.signal: Signal = None
        self.fiber: Fiber = None
        self.raman_amplifier: RamanAmplifier = None
        self.experiment: RamanSystem = None
        self.running = True
        self.command_buffer: list[Callable] = []
        self.start_time = time()

    def run(self):
        while self.running:
            log.info(f"Running for {time() - self.start_time}")
            sleep(0.1)

    def set_raman_amplifier(self, raman_amplifier: RamanAmplifier):
        self.raman_amplifier = raman_amplifier

    def set_signal(self, power: Power, wavelength: Length):
        self.signal = Signal()
        self.signal.power = power
        self.signal.wavelength = wavelength

    def set_fiber(self, fiber: Fiber, length: Length):
        self.fiber = fiber
        self.fiber.length = length

    def set_experiment(self):
        self.experiment = RamanSystem(self.fiber, self.signal, self.raman_amplifier)
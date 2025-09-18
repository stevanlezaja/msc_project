from raman_amplifier import RamanInputs, GainSpectrum
from custom_types import Frequency, PowerGain

from ..controller_base import _Controller

class PidController(_Controller):
    def __init__(self, p=0.2, i=0.1, d=0.1):
        self.p = p
        self.i = i
        self.d = d
        self.integral = 0.0
        self.e1 = 0.0

    def get_control(self, curr_input: RamanInputs, curr_output: GainSpectrum, target_output: GainSpectrum) -> RamanInputs:
        e = (target_output - curr_output).mean

        p = self.p * e

        i = self.integral + self.i * e
        if abs(i) < 1:
            self.integral = i

        d = self.d * (e - self.e1)
        self.e1 = e

        control = p  + i + d

        diff = RamanInputs()

        # new_input = curr_input - RamanInputs(p + i + d)
        # return new_input


if __name__ == "__main__":
    pid = PidController()

    initial = GainSpectrum()
    initial.spectrum = {
        Frequency(0, 'Hz'): PowerGain(10, ''),
        Frequency(1, 'Hz'): PowerGain(15, ''),
        Frequency(2, 'Hz'): PowerGain(20, ''),
        Frequency(3, 'Hz'): PowerGain(25, ''),
        Frequency(4, 'Hz'): PowerGain(7, ''),
    }

    target = GainSpectrum()
    target.spectrum = {
        Frequency(0, 'Hz'): PowerGain(1, ''),
        Frequency(1, 'Hz'): PowerGain(2, ''),
        Frequency(2, 'Hz'): PowerGain(3, ''),
        Frequency(3, 'Hz'): PowerGain(4, ''),
        Frequency(4, 'Hz'): PowerGain(1, ''),
    }

    curr_input = RamanInputs()


    for i in range(100):
        curr_input = pid.get_control(curr_input, initial, target)
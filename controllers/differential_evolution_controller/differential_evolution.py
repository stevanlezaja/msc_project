import scipy.optimize

import custom_types as ct
import raman_amplifier as ra

from ..controller_base import Controller

class DifferentialEvolutionController(Controller):
    def __init__(self):
        super().__init__()

    def get_control(self, curr_input: ra.RamanInputs, curr_output: ra.Spectrum[ct.Power], target_output: ra.Spectrum[ct.Power]) -> ra.RamanInputs:
        def optimization_fn(curr_input: ra.RamanInputs) -> float:
            return ra.mse(curr_output, target_output)
        result = scipy.optimize.differential_evolution(optimization_fn, bounds=[[0.25, 0.75], [1420, 1490]], maxiter=1)
        return ra.RamanInputs(powers=[ct.Power(float(result.x[0]), 'W')], wavelengths=[ct.Length(float(result.x[1]), 'nm')])

    def update_controller(self, error: ra.Spectrum[ct.Power], control_delta: ra.RamanInputs) -> None:
        return

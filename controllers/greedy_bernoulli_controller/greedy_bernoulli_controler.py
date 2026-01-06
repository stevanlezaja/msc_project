import copy
import torch
import matplotlib.axes
import numpy as np
import itertools

import raman_amplifier as ra
import custom_types as ct

from ..controller_base import Controller


class GreedyBernoulliController(torch.nn.Module, Controller):

    def __init__(self, *,
                 power_step: ct.Power = ct.Power(1, 'mW'),
                 wavelength_step: ct.Length = ct.Length(5, 'nm'),
                 lr: float = 1e-1,
                 weight_decay: float = 0.0,
                 beta: float = 1,
                 gamma: float = 0.99,
                 input_dim: int = 2,
                 ):
        super(torch.nn.Module, self).__init__()
        Controller.__init__(self)
        self._params['power_step'] = (ct.Power, power_step)
        self._params['wavelength_step'] = (ct.Length, wavelength_step)
        self._params['lr'] = (float, lr)
        self._params['weight_decay'] = (float, weight_decay)
        self._params['beta'] = (float, beta)
        self._params['gamma'] = (float, gamma)

        self.input_dim = input_dim
        self.logits = 0.01 * torch.randn(input_dim)
        self.best_reward = None
        self._baseline = 0.0
        self.history: dict[str, list[float]|dict[str, list[float]]] = {'probs': [], 'rewards': {'total': [], 'shape_loss': [], 'integral_loss': [], 'mse_loss': [], 'wavelength_spread': []}, 'baseline': []}
        self.avg_sample = torch.zeros_like(self.logits)
        self.prev_error: ra.Spectrum[ct.Power] | None = None
        self.output_integral: float = 0.0
        self.target_integral: float = 0.0
        self.last_sample = None
        self.last_reward = None

    @property
    def power_step(self) -> ct.Power:
        return self._params['power_step'][1]

    @property
    def wavelength_step(self) -> ct.Length:
        return self._params['wavelength_step'][1]

    @property
    def learning_rate(self) -> float:
        return self._params['lr'][1]

    @property
    def weight_decay(self) -> float:
        return self._params['weight_decay'][1]

    @property
    def beta(self) -> float:
        return self._params['beta'][1]

    @property
    def gamma(self) -> float:
        return self._params['gamma'][1]

    @property
    def rewards(self) -> list[float]:
        return self.history['rewards']['total']  # type: ignore

    @property
    def baseline(self) -> list[float]:
        return self.history['baseline']  # type: ignore

    def reward(
        self,
        curr_input: ra.RamanInputs,
        curr_output: ra.Spectrum[ct.Power],
        target_output: ra.Spectrum[ct.Power],
    ) -> float:

        def shape_difference(spec1: ra.Spectrum[ct.Power], spec2: ra.Spectrum[ct.Power]):
            int1 = ra.spectrum.integral(spec1).W
            scaled_spec1 = copy.deepcopy(spec1)

            int2 = ra.spectrum.integral(spec2).W
            scaled_spec2 = copy.deepcopy(spec2)

            difference_spec = scaled_spec1 / int1 - scaled_spec2 / int2

            return difference_spec.mean

        def integral_difference(spec1: ra.Spectrum[ct.Power], spec2: ra.Spectrum[ct.Power]):
            int_dif = ra.spectrum.integral(spec1).W - ra.spectrum.integral(spec2).W
            return int_dif if int_dif > 0 else - 10 * int_dif

        def wavelength_spread(wavelengths: list[ct.Length]):
            spread = 0
            for w1, w2 in itertools.combinations(wavelengths, 2):
                spread += abs(w1.nm - w2.nm) **0.5
            return spread

        sh_dif = 1 * shape_difference(curr_output, target_output)

        int_dif = integral_difference(curr_output, target_output)

        wl_spread = 0 * wavelength_spread(curr_input.wavelengths)

        self.history['rewards']['shape_loss'].append(sh_dif)  # type: ignore
        self.history['rewards']['integral_loss'].append(int_dif)  # type: ignore
        self.history['rewards']['wavelength_spread'].append(wl_spread)  # type: ignore

        loss = sh_dif + int_dif - wl_spread
        loss = ra.spectrum.mse(curr_output, target_output)

        # print(f"Reward is: {-loss}\n  Shape difference is {sh_dif/loss*100:.2f}%\n  Integral difference is {int_dif/loss*100:.2f}%\n")

        return -loss

    def get_control(
        self,
        curr_input: ra.RamanInputs,
        curr_output: ra.Spectrum[ct.Power],
        target_output: ra.Spectrum[ct.Power],
    ) -> ra.RamanInputs:
        self.curr_input = curr_input
        self.curr_output = curr_output
        self.target_output = target_output

        probs = torch.sigmoid(self.logits)
        self.history['probs'].append(probs.detach().numpy())  # type: ignore

        dist = torch.distributions.Bernoulli(probs)
        sample = dist.sample()

        self.last_sample = sample.detach()

        curr_reward = self.reward(curr_input, curr_output, target_output)

        if self.last_reward is not None and curr_reward < self.last_reward:
            zero_powers = [ct.Power(0.0, 'W') for _ in curr_input.powers]
            zero_wavelengths = [ct.Length(0.0, 'm') for _ in curr_input.wavelengths]
            return ra.RamanInputs(zero_powers, zero_wavelengths)

        power_sample = sample[:self.input_dim // 2]
        wavelength_sample = sample[self.input_dim // 2:]

        power_action = self.power_step.value * (power_sample * 2 - 1)
        wavelength_action = self.wavelength_step.value * (wavelength_sample * 2 - 1)

        powers_update = [ct.Power(float(p), 'W') for p in power_action]
        wavelength_update = [ct.Length(float(w), 'm') for w in wavelength_action]

        return ra.RamanInputs(powers_update, wavelength_update)

    def update_controller(
            self,
            error: ra.Spectrum[ct.Power],
            control_delta: ra.RamanInputs
        ) -> None:

        reward = self.reward(self.curr_input, self.curr_output, self.target_output)

        self.history['rewards']['total'].append(reward)  # type: ignore

        mse = ra.spectrum.mse(self.curr_output, self.target_output)
        self.history['rewards']['mse_loss'].append(mse)  # type: ignore

        self._baseline = self.gamma * self._baseline + (1 - self.gamma) * reward
        self.history['baseline'].append(self._baseline)  # type: ignore

        if not np.isfinite(reward):
            print("INFINITE REWARD")
            reward = -1e3   # or some strong penalty

        advantage = self.beta * (reward - self._baseline)

        self.prev_error = error

        sample = getattr(self, "last_sample", None)
        if sample is None:
            return

        probs = torch.sigmoid(self.logits)
        eligibility = sample - probs

        update = self.learning_rate * advantage * eligibility - self.weight_decay * self.logits
        self.logits += update
        if self.last_reward is not None:
            if reward > self.last_reward:
                self.last_reward = reward
        else:
            self.last_reward = reward

    def plot_custom_data(self, ax: matplotlib.axes.Axes):
        probs = np.array(self.history['probs'])  # shape: (steps, n_actions)
        # --- Step probability evolution ---
        ax.plot(probs[:, :])  # type: ignore
        ax.set_xlabel("Iteration")  # type: ignore
        ax.set_ylabel("Probability")  # type: ignore
        ax.set_title("Step probability evolution")  # type: ignore
        ax.grid()  # type: ignore
        ax.legend()  # type: ignore

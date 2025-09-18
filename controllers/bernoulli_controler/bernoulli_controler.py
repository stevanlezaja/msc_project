import torch

from src.raman_simulator import RaInputs, GainSpectrum, RamanSimulator

from ..controller_base import _Controller


class BernoulliController(torch.nn.Module, _Controller):
    def __init__(self, 
                 model: RamanSimulator, 
                 beta=1, 
                 weight_decay=1e-5,
                 input_dim=6, 
                 step=0.1, 
                 lr:float=1e-2, 
                 gamma=0, 
                 *args, 
                 **kwargs
                 ):
        super().__init__()
        self.model = model          # TODO proper intialization
        self.step = step
        self.logits = 0.2*torch.randn(input_dim)
        self.learning_rate = lr
        self.gamma = gamma
        self.beta = beta
        self.best_reward = None
        self.weight_decay = weight_decay
        self.baseline = 0.0
        self.history = {'probs': [], 'rewards': []}
        self.avg_sample = torch.zeros_like(self.logits)
        self.prev_error = None
    
    def get_control(self) -> RaInputs:
        probs = torch.sigmoid(self.logits)
        self.history['probs'].append(probs.detach().numpy())

        dist = torch.distributions.Bernoulli(probs)
        sample = dist.sample()
        action = self.step * (sample * 2 - 1)
        return RaInputs(action)

    def update_controller(
            self,
            error: GainSpectrum,
            x_delta: RaInputs
    ):
        sample = x_delta.value / self.step / 2 + 1
        loss = -1 * torch.norm(self.prev_error.value)**2 if self.prev_error is not None else 0.0
        reward = -loss

        reinforcement_factor = reward - self.baseline - self.beta
        self.baseline = self.gamma * self.baseline + (1 - self.gamma) * reward

        eligibility = sample - self.avg_sample
        self.avg_sample = self.gamma * self.avg_sample + (1 - self.gamma) * sample
        self.logits += torch.clamp(self.learning_rate * reinforcement_factor * eligibility - self.weight_decay * self.logits, min=-0.1, max=0.1)
        self.prev_error = error
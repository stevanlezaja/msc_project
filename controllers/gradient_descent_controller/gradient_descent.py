# import torch

# # from src.raman_simulator import RaInputs, GainSpectrum, RamanSimulator

# from ..controller_base import _Controller

# class GradientDescentController(_Controller):
#     def __init__(self, model: RamanSimulator, lr: float=1e-4, *args, **kwargs):
#         super().__init__(model)
#         self.learning_rate = lr

#     def get_control(
#             self, 
#             curr_input: RaInputs,
#             target_output: GainSpectrum,
#         ) -> RaInputs:

#         loss = torch.nn.MSELoss()
#         y = self.model(inputs=curr_input, detach_gradient=False)
#         current_cost = loss(y.value, target_output.value)
#         derivative = torch.autograd.grad(outputs=current_cost, inputs=curr_input.value, retain_graph=True)[0]
#         new_input = curr_input - RaInputs(curr_input.value * (self.learning_rate * derivative))
#         return new_input
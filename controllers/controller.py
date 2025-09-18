# from enum import Enum
# from abc import ABC, abstractmethod
# import torch

# # from src.raman_simulator import RamanSimulator, RaInputs, GainSpectrum

# # from . import GradientDescentController, BernoulliController, PolicyGradientControler, InverseController, PidController
# from .controller_base import _Controller


# class ControllerType(Enum):
#     GradientDescent = 'gradient_descent'
#     Bernoulli = 'bernoulli'
#     PolicyGradient = 'policy_gradient'
#     PID = 'pid'
#     Inverse = 'inverse'

#     @staticmethod
#     def from_string(type: str) -> "ControllerType":
#         try:
#             return ControllerType(type.lower())
#         except ValueError:
#             raise ValueError(f"Invalid controller type: {type}")


# class Controller(_Controller):
#     def __init__(self, model: RamanSimulator, controller_type: ControllerType, *args, **kwargs):
#         super().__init__(model)

#         if isinstance(controller_type, str):
#             controller_type = ControllerType.from_string(controller_type)
        
#         if not isinstance(controller_type, ControllerType):
#             raise ValueError(f"Controller Type must be a ControllerType Enum, got {controller_type.__class__.__name__}")

#         if controller_type == ControllerType.GradientDescent: self.controller = GradientDescentController(model=model, *args, **kwargs)
#         if controller_type == ControllerType.Bernoulli: self.controller = BernoulliController(model=model, *args, **kwargs)
#         if controller_type == ControllerType.PolicyGradient: self.controller = PolicyGradientControler(model=model, *args, **kwargs)
#         if controller_type == ControllerType.PID: self.controller = PidController(model=model, *args, **kwargs)
#         if controller_type == ControllerType.Inverse: self.controller = InverseController(*args, **kwargs)

#     def __call__(self, error: GainSpectrum = None) -> RaInputs:
#         return self.controller.get_control(error)

#     def get_control(self, error: GainSpectrum = None):
#         if isinstance(self.controller, BernoulliController):
#             control = self.controller.get_control()
#         else:
#             control = self.controller.get_control(error)

#         if not isinstance(control, RaInputs):
#             raise Exception("Oops soething went wrong creating the control")
#         return control


# def controller_step(controller: Controller, ra_sim: RamanSimulator, i: int, x_list: list[RaInputs], y_true: GainSpectrum, e_list: list[GainSpectrum]):
#     last_error = e_list[-1] if len(e_list) > 0 else None
#     x_delta = controller(last_error)
#     x = x_list[-1] + x_delta
#     error = y_true - ra_sim(inputs=x)
#     if isinstance(controller.controller, BernoulliController):
#         controller.controller.update_controller(error, x_delta)
#     x_list.append(x)
#     e_list[0, i] = torch.norm(error.value).item()
#     return x_list, e_list
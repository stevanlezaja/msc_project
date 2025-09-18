# import pytest
# import torch

# from src.raman_simulator import RamanSimulator, RaInputs, GainSpectrum

# from .controller import ControllerType, _Controller, Controller, GradientDescentController, \
#                         BernoulliController, PolicyGradientControler


# def test_from_string_valid():
#     assert ControllerType.from_string('gradient_descent') == ControllerType.GradientDescent
#     assert ControllerType.from_string('BERNOULLI') == ControllerType.Bernoulli
#     assert ControllerType.from_string('Policy_Gradient') == ControllerType.PolicyGradient

# def test_from_string_invalid():
#     with pytest.raises(ValueError, match="Invalid controller type: invalid_type"):
#         ControllerType.from_string('invalid_type')

#     with pytest.raises(ValueError, match="Invalid controller type:"):
#         ControllerType.from_string('')

#     with pytest.raises(ValueError, match="Invalid controller type:"):
#         ControllerType.from_string('GradientDescent')

# def test_controller_creation_with_enum():
#     model = RamanSimulator()

#     controller = Controller(model, ControllerType.GradientDescent)
#     assert isinstance(controller.controller, GradientDescentController)
    
#     controller = Controller(model, ControllerType.Bernoulli)
#     assert isinstance(controller.controller, BernoulliController)
    
#     controller = Controller(model, ControllerType.PolicyGradient)
#     assert isinstance(controller.controller, PolicyGradientControler)

# def test_controller_creation_with_string():
#     model = RamanSimulator()
    
#     controller = Controller(model, 'gradient_descent')
#     assert isinstance(controller.controller, GradientDescentController)
    
#     controller = Controller(model, 'bernoulli')
#     assert isinstance(controller.controller, BernoulliController)
    
#     controller = Controller(model, 'policy_gradient')
#     assert isinstance(controller.controller, PolicyGradientControler)

# def test_controller_creation_with_invalid_string():
#     model = RamanSimulator()
#     with pytest.raises(ValueError, match="Invalid controller type"):
#         Controller(model, "invalid_controller")

# def test_controller_creation_with_invalid_type():
#     model = RamanSimulator()
#     with pytest.raises(ValueError, match="Controller Type must be a ControllerType Enum"):
#         Controller(model, 12345)

# def test_controller_creation_with_mixed_case_string():
#     model = RamanSimulator()
#     controller = Controller(model, "PoLiCy_GrAdIeNt")
#     assert isinstance(controller.controller, PolicyGradientControler)

# def test_gradient_descent_control():
#     input = RaInputs(torch.rand(6,))
#     target = GainSpectrum(torch.rand(40,))
#     model = RamanSimulator()
#     controller = Controller(model, ControllerType.GradientDescent)
#     assert isinstance(controller.controller, GradientDescentController)
#     input_change = controller(input, target)
#     assert isinstance(input_change, RaInputs)


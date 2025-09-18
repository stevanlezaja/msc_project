# import numpy as np
# import torch
# import gym
# from gym import spaces

# from src.raman_simulator import RamanSimulator, RaInputs, GainSpectrum

# class RamanEnv(gym.Env):
#     def __init__(self, raman_model:RamanSimulator):
#         super(RamanEnv, self).__init__()
#         self.raman_model = raman_model
#         self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(6,), dtype=np.float32)
#         self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(20,), dtype=np.float32)

#     def reset(self):
#         self.state = torch.zeros(40)
#         return self.state

#     def step(self, action:RaInputs, target_output:GainSpectrum, threshold:float=1e-3):
#         action_scaled = self.scale_action(action)
#         outputs = self.raman_model.get_output(action_scaled)

#         reward = -np.linalg.norm(outputs.value.detach() - target_output.value.detach())

#         self.state = outputs
#         # if outputs.value - target_output.value < threshold:
#         #     done = True
#         # else:
#         #     done = False
#         done = False

#         return self.state, reward, done, {}

#     def scale_action(self, action):
#         return action


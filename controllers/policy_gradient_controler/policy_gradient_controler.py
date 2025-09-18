# import torch

# from src.raman_simulator import GainSpectrum, RaInputs, RamanSimulator
# # from .learning_env import RamanEnv
# from .policy_network import PolicyNetwork
# from .policy_gradient import PolicyGradient
# from ..controller_base import _Controller

# class PolicyGradientControler(_Controller):
#     def __init__(self, model:RamanSimulator):
#         self.policy = PolicyNetwork()
#         self.agent = PolicyGradient(self.policy)
#         self.model = model

#     def train(
#             self, 
#             state:GainSpectrum, 
#             target:GainSpectrum, 
#             num_episodes:int=1, 
#             steps_per_episode:int=100, 
#             lr:float=1e-4
#             ) -> None:
        
#         optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)

#         for _ in range(num_episodes):
#             # state = GainSpectrum(self.env.reset())
#             total_reward = 0
#             log_probs = []
#             rewards = []

#             ra_input = None

#             for _ in range(steps_per_episode):
#                 action, log_prob = self.agent.select_action(state, target)
#                 if ra_input is None:
#                     ra_input = action
#                 else:
#                     ra_input += action
#                 next_state = self.model.get_output(ra_input)

#                 reward = -sum((next_state - target).value.detach().numpy()**2)

#                 log_probs.append(log_prob)
#                 rewards.append(reward)

#                 state = next_state
#                 total_reward += reward

#             self.agent.update(log_probs, rewards, optimizer)
#             # print(f"Episode {episode}, Total Reward: {total_reward}")

#     def get_control(self, state, target, action_std=0.1):
#         return self.agent.select_action(state, target, action_std)

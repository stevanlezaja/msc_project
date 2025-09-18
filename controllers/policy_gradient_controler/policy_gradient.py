# import torch
# import torch.optim as optim

# from src.raman_simulator import RaInputs, GainSpectrum

# from .policy_network import PolicyNetwork

# class PolicyGradient:
#     def __init__(self, policy:PolicyNetwork, lr:float=1e-3):
#         self.policy = policy
#         self.optimizer = optim.Adam(policy.parameters(), lr=lr)
#         self.gamma = 0.8

#     def select_action(self, state:GainSpectrum, target:GainSpectrum, action_std:float) -> tuple[RaInputs, float]:

#         error = target.value - state.value
#         action = self.policy(error)
#         dist = torch.distributions.Normal(action, action_std)
#         sampled_action = RaInputs(dist.sample())
#         log_prob = dist.log_prob(sampled_action.value).sum()

#         return sampled_action, log_prob

#     def update(self, log_probs, rewards, optimizer):
#         discounted_rewards = []
#         R = 0
#         for r in reversed(rewards):
#             R = r + self.gamma * R
#             discounted_rewards.insert(0, R)
        
#         discounted_rewards = torch.tensor(discounted_rewards, dtype=torch.float32)
#         discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-9)

#         policy_loss = []
#         for log_prob, reward in zip(log_probs, discounted_rewards):
#             policy_loss.append(-log_prob * reward)
        
#         optimizer.zero_grad()
#         loss = torch.stack(policy_loss).sum()
#         loss.backward()
#         optimizer.step()

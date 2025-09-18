# import torch
# import torch.nn as nn

# class PolicyNetwork(nn.Module):
#     def __init__(self, path=None, input_dim=40, output_dim=6):
#         super(PolicyNetwork, self).__init__()
#         self.fc1 = nn.Linear(input_dim, 128)
#         self.fc2 = nn.Linear(128, 128)
#         self.fc3 = nn.Linear(128, output_dim)
#         self.tanh = nn.ReLU()
#         if path == None:
#             self.load_state_dict(torch.load('rl_model/model.pth'))
#         elif path != 'new_model':
#             self.load_state_dict(torch.load(f'rl_model/{path}'))


#     def forward(self, x):
#         x = torch.relu(self.fc1(x))
#         # x = torch.relu(self.fc2(x))
#         x = self.tanh(self.fc3(x))
#         return x

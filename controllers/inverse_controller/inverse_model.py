# import os
# import torch
# import torch.nn as nn
# import torch.optim as optim

# from src.utils import load_data
# from src.raman_simulator import RaInputs, GainSpectrum
# from src.raman_simulator.forward_nn import SimpleNet

# from ..controller_base import _Controller

# class InverseController(SimpleNet, _Controller):
#     def __init__(self,
#                  lr=0.001,
#                  num_epochs=100,
#                  batch_size=64,
#                  optimizer_type='adam',
#                  loss_fn='mse',
#                  l2_lambda=0.0,
#                  model_path='inverse_model/inverse_model.pt',
#                  device=None,
#                  visualize=False,
#                  *args,
#                  **kwargs):
        
#         super().__init__(input_size=40, output_size=6, layer1_hu=50, layer2_hu=50, weight_init='normal', activation='relu')

#         os.makedirs(os.path.dirname(model_path), exist_ok=True)
#         self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
#         self.to(self.device)

#         self.model_path = model_path
#         self.visualize = visualize

#         if os.path.isfile(model_path):
#             print(f"Loading model from {model_path}")
#             self.load_state_dict(torch.load(model_path, map_location=self.device))
#             self.to(self.device)
#         else:
#             Y_train, X_train, Y_test, X_test, _ = load_data('data/data.mat')
#             print(f"No existing model found. Training a new model and saving to {model_path}")
#             self.train_and_save(X_train, Y_train, lr, num_epochs, batch_size, optimizer_type, loss_fn, l2_lambda)

#     def train_and_save(self, X_train, y_train, lr, num_epochs, batch_size, optimizer_type, loss_fn, l2_lambda):
#         # Ensure that X_train and y_train are the .value attributes
#         X_train = torch.stack([item.value for item in X_train]).to(self.device)
#         y_train = torch.stack([item.value for item in y_train]).to(self.device)
        
#         # Create the DataLoader
#         train_loader = self.train_loader(X_train, y_train)
        
#         if loss_fn == 'mse':
#             criterion = nn.MSELoss()
#         elif loss_fn == 'crossentropy':
#             criterion = nn.CrossEntropyLoss()
#         else:
#             raise ValueError(f"Unsupported loss function: {loss_fn}")
        
#         if optimizer_type == 'adam':
#             optimizer = optim.Adam(self.parameters(), lr=lr)
#         elif optimizer_type == 'sgd':
#             optimizer = optim.SGD(self.parameters(), lr=lr)
#         else:
#             raise ValueError(f"Unsupported optimizer type: {optimizer_type}")

#         # Train the model
#         self.train(
#             train_loader=train_loader,
#             criterion=criterion,
#             optimizer=optimizer,
#             num_epochs=num_epochs,
#             device=self.device,
#             visualize=self.visualize,
#             l2_lambda=l2_lambda
#         )

#         # Save the trained model
#         torch.save(self.state_dict(), self.model_path)
#         print(f"Model saved to {self.model_path}")

#     def get_control(self, curr_input, target_output: GainSpectrum):
#         y = target_output.value.to(self.device)
#         self.eval()
#         with torch.no_grad():
#             predicted_input = self(y).cpu()
#             return RaInputs(predicted_input)

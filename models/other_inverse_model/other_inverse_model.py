import os
import random
from copy import deepcopy

import torch
import torch.nn as nn
import torch.optim as optim

import custom_types as ct
import raman_amplifier as ra
from utils.loading_data_from_file import load_raman_dataset

from ..simple_net import SimpleNet


def load_data(path: str, test_ratio: float = 0.2, seed: int = 42):
    def _compute_spectrum_norm(dataset_path):
        min_val = float('inf')
        max_val = float('-inf')
        for _, spectrum in load_raman_dataset(dataset_path):
            arr = deepcopy(spectrum).as_array()
            min_val = min(min_val, arr.min())
            max_val = max(max_val, arr.max())
        return min_val, max_val
    norm_min, norm_max = _compute_spectrum_norm(path)
    ra.Spectrum.norm_min = norm_min
    ra.Spectrum.norm_max = norm_max
    pairs = list(load_raman_dataset(path))

    rng = random.Random(seed)
    rng.shuffle(pairs)

    split = int(len(pairs) * (1 - test_ratio))
    train_pairs = pairs[:split]
    test_pairs  = pairs[split:]

    def to_tensor(pairs):
        X = torch.stack([
            torch.tensor(ri.normalize().as_array(), dtype=torch.float32)
            for ri, _ in pairs
        ])
        Y = torch.stack([
            torch.tensor(sp.normalize().as_array(), dtype=torch.float32)
            for _, sp in pairs
        ])
        return X, Y

    X_train, Y_train = to_tensor(train_pairs)
    X_test,  Y_test  = to_tensor(test_pairs)

    return X_train, Y_train, X_test, Y_test


class OtherInverseModel(SimpleNet):
    def __init__(self,
                 lr=0.001,
                 num_epochs=100,
                 batch_size=64,
                 optimizer_type='adam',
                 loss_fn='mse',
                 l2_lambda=0.0,
                 model_path='inverse_model/inverse_model.pt',
                 device=None,
                 visualize=False,
                 *args,
                 **kwargs):
        
        super().__init__(input_size=40, output_size=6, layer1_hu=50, layer2_hu=50, weight_init='normal', activation='relu')

        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

        self.model_path = model_path
        self.visualize = visualize

        if os.path.isfile(model_path):
            print(f"Loading model from {model_path}")
            self.load_state_dict(torch.load(model_path, map_location=self.device))
            self.to(self.device)
        else:
            Y_train, X_train, Y_test, X_test = load_data('data/raman_simulator/3_pumps/100_fiber_0.0_ratio_sorted.json')
            print(f"No existing model found. Training a new model and saving to {model_path}")
            self.train_and_save(X_train, Y_train, lr, num_epochs, batch_size, optimizer_type, loss_fn, l2_lambda)

    def train_and_save(self, X_train, y_train, lr, num_epochs, batch_size, optimizer_type, loss_fn, l2_lambda):
        X_train = torch.stack([item.values for item in X_train]).to(self.device)
        y_train = torch.stack([item.values for item in y_train]).to(self.device)
        
        train_loader = self.train_loader(X_train, y_train)
        
        if loss_fn == 'mse':
            criterion = nn.MSELoss()
        elif loss_fn == 'crossentropy':
            criterion = nn.CrossEntropyLoss()
        else:
            raise ValueError(f"Unsupported loss function: {loss_fn}")
        
        if optimizer_type == 'adam':
            optimizer = optim.Adam(self.parameters(), lr=lr)
        elif optimizer_type == 'sgd':
            optimizer = optim.SGD(self.parameters(), lr=lr)
        else:
            raise ValueError(f"Unsupported optimizer type: {optimizer_type}")

        self.train(
            train_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            num_epochs=num_epochs,
            device=self.device,
            visualize=self.visualize,
            l2_lambda=l2_lambda
        )

        torch.save(self.state_dict(), self.model_path)
        print(f"Model saved to {self.model_path}")

    def get_raman_inputs(self, target_output: ra.Spectrum[ct.Power]):
        y = torch.tensor(target_output.normalize().as_array(), dtype=torch.float32).to(self.device)
        self.eval()
        with torch.no_grad():
            predicted_input = self(y).cpu().detach().numpy()
            return ra.RamanInputs.from_array(predicted_input).denormalize()

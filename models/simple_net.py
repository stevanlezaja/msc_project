import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.adadelta
import torch.optim.adam
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import networkx as nx
import numpy as np

class SimpleNet(nn.Module):
    def __init__(self, input_size, output_size, layer1_hu=0, layer2_hu=0, weight_init='normal', activation='relu'):
        super(SimpleNet, self).__init__()
        if layer1_hu != 0 and layer2_hu != 0:
            self.fc = nn.Sequential(
                nn.Linear(input_size, layer1_hu),
                nn.ReLU() if activation == 'relu' else nn.Sigmoid(),
                nn.Linear(layer1_hu, layer2_hu),
                nn.ReLU() if activation == 'relu' else nn.Sigmoid(),
                nn.Linear(layer2_hu, output_size),
            )
        elif layer1_hu != 0:
            self.fc = nn.Sequential(
                nn.Linear(input_size, layer1_hu),
                nn.ReLU() if activation == 'relu' else nn.Sigmoid(),
                nn.Linear(layer1_hu, output_size),
            )
        else:
            self.fc = nn.Linear(input_size, output_size)

        self.input_size = input_size
        self.output_size = output_size
        self.batch_size = 64
        self._initialize_weights(weight_init)

    def _initialize_weights(self, method):
        if method == 'uniform':
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.uniform_(m.bias)
        elif method == 'normal':
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight)
                    if m.bias is not None:
                        nn.init.normal_(m.bias)
        elif method == 'xavier':
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_normal_(m.weight)
                    if m.bias is not None:
                        nn.init.xavier_normal_(m.bias)
        else:
            raise ValueError(f"Unknown weight initialization method: {method}")

    def train_loader(self, X_train, y_train):
        train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        return train_loader

    def forward(self, x):
        out = self.fc(x)
        return out

    def train_model(self, train_loader, criterion=torch.nn.MSELoss, optimizer=torch.optim.Adam, num_epochs=100, device='cpu', visualize=False, l2_lambda=0.0):
        if visualize:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            plt.ion()
            fig.show()
            fig.canvas.draw()

        loss_history = []

        for _ in tqdm(range(num_epochs)):
            running_loss = 0.0
            for inputs, targets in train_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = self.forward(inputs)
                loss = criterion(outputs, targets)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

            epoch_loss = running_loss / len(train_loader)
            loss_history.append(epoch_loss)

            if visualize:
                ax1.clear()
                draw_neural_network(self, ax1)
                plot_loss(ax2, loss_history)
                fig.canvas.draw()
                plt.pause(0.0001)

def plot_loss(ax, loss_history):
    """ Helper function to plot the loss over time """
    ax.clear()
    # ax.set_ylim([0, loss_history[-1] * 10])
    ax.plot(loss_history, label='Training Loss', color='blue')
    ax.set_title("Training Loss Over Epochs")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()

def draw_neural_network(model, ax):
    """ Function to visualize the architecture of the neural network """
    G = nx.DiGraph()

    input_nodes = [f"Input {i}" for i in range(model.input_size)]
    for node in input_nodes:
        G.add_node(node, layer="input")

    hidden_nodes = []
    if isinstance(model.fc, nn.Sequential):
        hidden_layers = [layer for layer in model.fc if isinstance(layer, nn.Linear)]
        for layer_idx, layer in enumerate(hidden_layers[:-1]):
            layer_nodes = [f"Hidden {layer_idx} Neuron {i}" for i in range(layer.out_features)]
            hidden_nodes.append(layer_nodes)
            for node in layer_nodes:
                G.add_node(node, layer=f"hidden_{layer_idx}")

    output_nodes = [f"Output {i}" for i in range(model.output_size)]
    for node in output_nodes:
        G.add_node(node, layer="output")

    # Add edges with weights
    layers = [input_nodes] + hidden_nodes + [output_nodes]
    for layer_idx in range(len(layers) - 1):
        current_layer = layers[layer_idx]
        next_layer = layers[layer_idx + 1]
        weights = model.fc[layer_idx * 2].weight.data.cpu().numpy() if isinstance(model.fc, nn.Sequential) else model.fc.weight.data.cpu().numpy()
        for i, current_node in enumerate(current_layer):
            for j, next_node in enumerate(next_layer):
                weight = weights[j, i]
                G.add_edge(current_node, next_node, weight=weight)

    # Position nodes
    pos = {}
    layer_dist = 2
    for layer_idx, layer in enumerate(layers):
        y_dist = 10 / (len(layer) - 1) if len(layer) > 1 else 0
        for i, node in enumerate(layer):
            pos[node] = (layer_idx * layer_dist, -i * y_dist)

    edge_weights = [G[u][v]['weight'] for u, v in G.edges()]
    normalized_weights = (edge_weights - np.min(edge_weights)) / (np.max(edge_weights) - np.min(edge_weights))
    colors = cm.plasma(normalized_weights)  # Use a colormap of your choice (viridis, plasma, etc.)
    edge_thickness = [abs(weight) * 5 for weight in edge_weights]  # Scale for better visibility

    nx.draw(G, pos, with_labels=False, node_size=100, node_color='skyblue', arrows=False,
            edge_color=colors, width=edge_thickness, ax=ax, labels={node: node for node in G.nodes()})
    ax.set_title("Neural Network Architecture")    

def visualize_test(Y, Y_pred, model_name):
    plt.figure(figsize=(8, 6))
    y_min = np.inf
    y_max = -np.inf
    colors = plt.cm.get_cmap('tab10', 10)
    for i in range(10):
        idx = np.random.randint(0, Y.shape[0])
        y_min = min(y_min, np.min(Y[idx]), np.min(Y_pred[idx]))
        y_max = max(y_max, np.max(Y[idx]), np.max(Y_pred[idx]))
        color = colors(i % 10)
        plt.ylim(min(0, 1.1*y_min), max(0, 1.1*y_max))
        plt.plot(Y[idx], linestyle='--', color=color, label='True')
        plt.plot(Y_pred[idx], linestyle='-', color=color, label='Predicted')
    plt.savefig(f"plots/{model_name}.png")
    plt.close()

def parse_model_params(model_name):
    parts = model_name.split("_")
    layer1_hu = int(parts[1].replace("hu1", ""))
    layer2_hu = int(parts[2].replace("hu2", ""))
    ensemble_size = int(parts[4].replace("models", ""))
    return layer1_hu, layer2_hu, ensemble_size

def train_ensemble_model(X, Y, ensemble_size, epochs, lr, optimizer_name, layer1_hu, layer2_hu, weight_init, activation, visualize=False, best=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device {device}")
    print(f"Training model with hidden units={layer1_hu}:{layer2_hu}, {ensemble_size} models, {epochs} epochs, lr={lr}, optimizer={optimizer_name}, weight_init={weight_init}")

    model_name = f"model_{layer1_hu}hu1_{layer2_hu}hu2_{activation}_{ensemble_size}models_{weight_init}_epochs{epochs}_lr{lr}_{optimizer_name}"
    if best:
        model_name = f"best{model_name}"

    for idx in range(ensemble_size):
        X_train, Y_train = X.to(device), Y.to(device)
        model = SimpleNet(X_train.shape[1], Y_train.shape[1], 
                          layer1_hu=layer1_hu, 
                          layer2_hu=layer2_hu, 
                          weight_init=weight_init).to(device)

        if optimizer_name == 'Adam':
            optimizer = optim.Adam(model.parameters(), lr=lr)
        elif optimizer_name == 'SGD':
            optimizer = optim.SGD(model.parameters(), lr=lr)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer}")
        
        criterion = nn.MSELoss()

        model.train(model.train_loader(X_train, Y_train), criterion, optimizer, epochs, device, visualize)
        if ensemble_size > 1:
            curr_model_name = f"{model_name}_{idx}"
        else:
            curr_model_name = model_name
        torch.save(model.state_dict(), f"models/{curr_model_name}")

def test_model(X, Y, model_name, visualize=True):
    layer1_hu, layer2_hu, ensemble_size = parse_model_params(model_name)
    Y_pred_list = []
    for idx in range(ensemble_size):
        if ensemble_size > 1:
            curr_model_name = f"{model_name}_{idx}"
        else:
            curr_model_name = model_name

        model = SimpleNet(input_size=X.shape[1], output_size=Y.shape[1], layer1_hu=layer1_hu, layer2_hu=layer2_hu)
        state_dict = torch.load(f"models/{curr_model_name}", weights_only=True)

        new_state_dict = {}
        for key in model.state_dict().keys():
            if key in state_dict:
                saved_param = state_dict[key]
                current_param = model.state_dict()[key]
                
                if saved_param.shape == current_param.shape:
                    new_state_dict[key] = saved_param
                else:
                    print(f"Skipping loading parameter {key} due to size mismatch: {saved_param.shape} vs {current_param.shape}")
            else:
                print(f"Missing parameter {key} in saved state_dict.")

        model.load_state_dict(new_state_dict, strict=False)

        Y_pred = model(X)
        loss = nn.MSELoss()(Y_pred, Y)
        Y_pred_list.append(Y_pred)
    Y_pred = torch.mean(torch.stack(Y_pred_list), dim=0)
    loss = nn.MSELoss()(Y_pred, Y)
    print(f"Test loss: {loss.item()}")

    if visualize:
        visualize_test(Y.detach().numpy(), Y_pred.detach().numpy(), model_name)
    return Y_pred
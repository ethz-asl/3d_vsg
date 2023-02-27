import torch


class SimpleMLP(torch.nn.Module):
    """Implementation of a simple Multilayer Perception for variability prediction"""
    def __init__(self, n_node_features, n_out_classes, hidden_layers):
        super().__init__()
        self.fc1 = torch.nn.Linear(n_node_features, hidden_layers[0])
        self.relu1 = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(hidden_layers[0], hidden_layers[1])
        self.relu2 = torch.nn.ReLU()
        self.fc3 = torch.nn.Linear(hidden_layers[1], n_out_classes)
        self.relu3 = torch.nn.ReLU()

    def forward(self, x_in):
        x1 = self.relu1(self.fc1(x_in))
        x2 = self.relu2(self.fc2(x1))
        x_out = self.relu3(self.fc3(x2))
        return x_out

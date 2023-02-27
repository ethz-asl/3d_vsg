import torch
from torch_geometric.nn import NNConv


class SimpleMPGNN(torch.nn.Module):
    """Implementation of main proposed architecture for DeltaVSG, a 2 layer Message-Passing Graph Convolution Neural
    Network"""
    def __init__(self, n_node_features, n_out_classes, n_edge_features, hidden_layers):
        super().__init__()
        n_i = n_edge_features
        n_o = n_node_features * hidden_layers[0]
        n_h = int((n_i + n_o)/2)
        mlp_1 = torch.nn.Sequential(torch.nn.Linear(n_i, n_h), torch.nn.ReLU(), torch.nn.Linear(n_h, n_o))
        self.conv1 = NNConv(n_node_features, hidden_layers[0], mlp_1)

        n_i = n_edge_features
        n_o = hidden_layers[0] * n_out_classes
        n_h = int((n_i + n_o)/2)
        mlp_2 = torch.nn.Sequential(torch.nn.Linear(n_i, n_h), torch.nn.ReLU(), torch.nn.Linear(n_h, n_o))
        self.conv2 = NNConv(hidden_layers[0], n_out_classes, mlp_2)
        self.relu1 = torch.nn.LeakyReLU()

    def forward(self, x_in, edge_index, edge_atts):
        x1 = self.relu1(self.conv1(x_in, edge_index, edge_atts))
        x2 = self.conv2(x1, edge_index, edge_atts)
        return x2

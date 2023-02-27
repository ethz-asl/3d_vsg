import torch
from torch_geometric.nn import TransformerConv


class PPNBaseline(torch.nn.Module):
    """
    Implementation of the Proximity Prediction Network from "Spatial Commonsense Graph for Object Localisation in
    Partial Scenes (Giuliari et. al)", i.e. a 2 layer Graph Transformer Network, adapted for Variability prediction """

    def __init__(self, n_node_features, n_out_classes, n_edge_features, hidden_layers):
        super().__init__()
        self.transformer1 = TransformerConv(n_node_features, hidden_layers[0], edge_dim=n_edge_features, beta=True, dropout=0.2)
        self.transformer2 = TransformerConv(hidden_layers[0], hidden_layers[1], edge_dim=n_edge_features, beta=True, dropout=0.2)
        self.relu1 = torch.nn.LeakyReLU()
        self.relu2 = torch.nn.LeakyReLU()
        self.mlp = torch.nn.Linear(hidden_layers[0] + hidden_layers[1], n_out_classes)

    def forward(self, x_in, edge_index, edge_atts):
        x1 = self.relu1(self.transformer1(x_in, edge_index, edge_atts))
        x2 = self.relu2(self.transformer2(x1, edge_index, edge_atts))
        out = self.mlp(torch.cat((x1, x2), dim=1))
        return out

import torch


class SelfAttentionLayer(torch.nn.Module):
    """Basic implementation of a self-attention layer"""
    def __init__(self, d_in, d_out):
        super().__init__()
        self.d_in = d_in
        self.d_out = d_out
        self.value_net = torch.nn.Linear(d_in, d_out, bias=False)
        self.query_net = torch.nn.Linear(d_in, d_in, bias=False)
        self.key_net = torch.nn.Linear(d_in, d_in, bias=False)

    def forward(self, input):
        Q = self.query_net(input)
        V = self.value_net(input)
        K = self.key_net(input)

        W = torch.nn.functional.softmax(Q@K.T, dim=-1)

        return W@V


class SimpleSA(torch.nn.Module):
    """Implementation of a simple multilayer perception with self-attention, as proposed in
    LayoutTransformer (Gupta et al.)"""
    def __init__(self, n_node_features, n_out_classes):
        super().__init__()
        self.sa1 = SelfAttentionLayer(n_node_features, n_node_features)
        self.fc1 = torch.nn.Linear(n_node_features, n_node_features)
        self.relu1 = torch.nn.ReLU()
        self.relu2 = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(n_node_features, n_out_classes)
        self.bn1 = torch.nn.BatchNorm1d(n_node_features)
        self.bn2 = torch.nn.BatchNorm1d(n_node_features)

    def forward(self, x_in):
        x1 = self.relu1(self.bn1(self.sa1(x_in) + x_in))
        x2 = self.relu2(self.bn2(self.fc1(x_in) + x1))
        x_out = self.fc2(x2)
        return x_out

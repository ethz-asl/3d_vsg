import torch


class FocalLoss(torch.nn.Module):
    """Implementation of a Focal Loss (Lin et al.)"""
    def __init__(self, gamma=1, weights=(1, 1, 1)):
        super().__init__()
        self.gamma = gamma
        self.weights = weights

    def forward(self, pred, label):
        # Convert logits to probabilities
        prob = torch.sigmoid(pred)
        # Calculate Binary Cross Entropy Loss
        label_prob = torch.clip(torch.mul(label, prob) + torch.mul(1-label, 1-prob), 0.0000001, 0.999999)
        bce_loss = -torch.log(label_prob)

        # Apply focal term
        focal_loss = torch.mul(torch.pow(1-label_prob, self.gamma), bce_loss)
        weights = torch.mul(label, self.weights) + (1 - label)
        weighted = torch.mul(weights, focal_loss)
        return weighted

import pickle
from collections import OrderedDict
import torch


def set_weights(model: torch.nn.Module, weights_file: str):
    """Set weights of a pytorch model using the weights from a file.

    Args:
        model (torch.nn.Module): Model.
        weights_file (str): Path to weights file.
    """

    with open(weights_file, "rb") as file:
        weights = pickle.load(file)

    state_dict = OrderedDict(
        {k: torch.Tensor(v) for k, v in zip(model.state_dict().keys(), weights)}
    )

    model.load_state_dict(state_dict, strict=True)


class WeightedL1Loss(torch.nn.Module):
    def __init__(self):
        super(WeightedL1Loss, self).__init__()

    def forward(self, input, target, weight, dim=None):
        return torch.mean(weight * torch.abs(input - target), dim=dim)


class WeightedL2Loss(torch.nn.Module):
    def __init__(self):
        super(WeightedL2Loss, self).__init__()

    def forward(self, input, target, weight, dim=None):
        return torch.mean(weight * (input - target) ** 2, dim=dim)

import torch
import torch.nn as nn


class PeriodicFeaturesCosSin(nn.Module):
    """
    Converts a specified part of the input to periodic features by
    replacing those features f with two features sin(scale * f), w2 * cos(scale * f).
    """

    def __init__(self, ndim, ind, scale=1.0):
        """Constructor

        Args:
          ndim (int): number of dimensions
          ind (iterable): indices of input elements to convert to periodic features
          scale: Scalar or iterable, used to scale inputs before converting them to periodic features
        """

        super(PeriodicFeaturesCosSin, self).__init__()

        self.ndim = ndim
        if torch.is_tensor(ind):
            self.register_buffer("ind", torch._cast_Long(ind))
        else:
            self.register_buffer("ind", torch.tensor(ind, dtype=torch.long))

        ind_ = []
        for i in range(self.ndim):
            if not i in self.ind:
                ind_ += [i]
        self.register_buffer("ind_", torch.tensor(ind_, dtype=torch.long))

        if torch.is_tensor(scale):
            self.register_buffer("scale", scale)
        else:
            self.scale = scale

    def forward(self, inputs):
        inputs_ = inputs[..., self.ind]
        inputs_ = self.scale * inputs_

        inputs_ = torch.cat(
            (
                torch.sin(inputs_),
                torch.cos(inputs_),
            ),
            -1,
        )
        out = torch.cat((inputs_, inputs[..., self.ind_]), -1)
        return out

        # inputs_ = self.weights[:, 0] * torch.sin(inputs_) + self.weights[
        #    :, 1
        # ] * torch.cos(inputs_)
        # if self.apply_bias:
        #    inputs_ = inputs_ + self.bias
        # inputs_ = self.activation(inputs_)
        # out = torch.cat((inputs_, inputs[..., self.ind_]), -1)
        # return out[..., self.inv_perm]

import torch
import torch.nn as nn
from main.models.flow_base import ConditionalFlowBase
from normflows.distributions.base import UniformGaussian


class MLP(nn.Module):
    def __init__(self, dimensions=[1, 128, 1]):
        super(MLP, self).__init__()

        layers = []

        for i in range(len(dimensions) - 1):
            layers.append(nn.Linear(dimensions[i], dimensions[i + 1]))
            if i < len(dimensions) - 2:
                layers.append(nn.Sigmoid())

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class GaussianFlow1D(ConditionalFlowBase):
    def __init__(
        self,
        q0: UniformGaussian,
        subnet_dimensions=[1, 128, 1],
        standard_scaler_dataset: torch.Tensor = None,
    ):
        super().__init__(q0)

        self.NN_0 = MLP(dimensions=subnet_dimensions)
        self.NN_1 = MLP(dimensions=subnet_dimensions)

        self.q0 = q0

        if standard_scaler_dataset is not None:
            self.scaler_m = torch.mean(
                standard_scaler_dataset, dim=0, keepdim=True
            ).cuda()
            self.scaler_s = torch.std(
                standard_scaler_dataset, dim=0, keepdim=True, unbiased=False
            ).cuda()
        else:
            self.scaler_m = None
            self.scaler_s = None

    def forward_and_log_det(self, z, context=None):
        if self.scaler_m is not None and self.scaler_s is not None:
            context = (context - self.scaler_m) / self.scaler_s

        NN_0_output = self.NN_0(context)
        NN_1_output = self.NN_1(context)
        x = z * NN_0_output + NN_1_output

        log_jac_det = torch.log(NN_0_output.abs())[:, 0]
        return x, log_jac_det

    def inverse_and_log_det(self, x_int, context):
        if self.scaler_m is not None and self.scaler_s is not None:
            context = (context - self.scaler_m) / self.scaler_s

        NN_0_output = self.NN_0(context)
        NN_1_output = self.NN_1(context)
        z = (x_int - NN_1_output) / NN_0_output

        log_jac_det = -torch.log(NN_0_output.abs())[:, 0]
        return z, log_jac_det


if __name__ == "__main__":
    q0 = UniformGaussian(ndim=1, ind=[])
    flow = GaussianFlow1D(q0=q0, subnet_dimensions=[1, 128, 1])

    z = torch.randn(1000, 1)
    context = torch.randn(1000, 1)

    x, log_jac_det = flow.forward_and_log_det(z=z, context=context)
    z_1, log_jac_det = flow.inverse(x_int=x, context=context)

    print(torch.isclose(z, z_1).all())

    # Count flow parameters
    num_flow_params = 0

    for param in flow.parameters():
        num_flow_params += param.numel()

    print(num_flow_params)

from FrEIA.framework import ReversibleGraphNet, SequenceINN
from FrEIA.modules import RNVPCouplingBlock, PermuteRandom
import torch.nn as nn
import torch
from typing import List
from normflows.distributions.base import UniformGaussian
from main.models.flow_base import ConditionalFlowBase
from normflows.distributions.base import UniformGaussian


class RNVPFrEIAFlow(ConditionalFlowBase):
    def __init__(self, q0: UniformGaussian, freia_flow: ReversibleGraphNet):
        """Wrapper for the RealNVP FrEIA flows.

        Args:
          q0: Base distribution
          freia_flow: FrEIA flow
        """

        super().__init__(q0)
        self.q0 = q0
        self.freia_flow = freia_flow

    def forward_and_log_det(self, z, context=None):
        """Transforms latent variable z to the variable x_int and
        computes log determinant of the Jacobian

        Args:
          z: Batch in the latent space
          context: Batch of conditional variables

        Returns:
          Batch in the space of the target distribution,
          log determinant of the Jacobian
        """

        return self.freia_flow(z, (context,))

    def inverse_and_log_det(self, x_int, context=None):
        """Transforms flow variable x_int to the latent variable z and
        computes log determinant of the Jacobian

        Args:
          x_int: Batch in the space of the target distribution
          context: Batch of conditional variables

        Returns:
          Batch in the latent space,
          log determinant of the Jacobian
        """

        return self.freia_flow(x_int, (context,), rev=True)


def create_conditional_RNVP_flow(
    ndim_in: int = 1,
    ndim_cond: int = 1,
    subnets_dimensionality: List[int] = [512],
    NO_blocks: int = 4,
    device: str = "cuda",
    seed: int = None,
) -> torch.nn.Module:
    """Get a conditional invertible model based on the RealNVP architecture.

    Args:
        ndim_in (int, optional): Number of input dimensions. Defaults to 1.
        ndim_cond (int, optional): Number of conditional dimensions. Defaults to 1.
        subnets_dimensionality (list, optional): Dimensionality of the t_i and s_i subnets of
            the RealNVP coupling blocks (list of hidden layers). Defaults to [512].
        NO_blocks (int, optional): Number of coupling blocks. Defaults to 4.
        device (str, optional): Device to use. Defaults to "cuda".
        seed (int, optional): Seed. Defaults to None.

    Returns:
        torch.nn.Module: Invertible model.
    """

    if seed is not None:
        previous_state_cpu = torch.random.get_rng_state()
        previous_state_gpu = torch.cuda.get_rng_state()

        torch.manual_seed(seed)

    def subnet_fc(c_in, c_out):
        layers = []
        for c in subnets_dimensionality:
            layers += [nn.Linear(c_in, c), nn.ReLU()]
            c_in = c
        layers += [nn.Linear(c_in, c_out)]
        return nn.Sequential(*layers)

    inn = SequenceINN(ndim_in)

    for k in range(NO_blocks):
        inn.append(
            RNVPCouplingBlock,
            subnet_constructor=subnet_fc,
            cond_shape=(ndim_cond,),
            cond=0,
        )
        inn.append(PermuteRandom, seed=k)

    inn.to(device)

    if seed is not None:
        torch.random.set_rng_state(previous_state_cpu)
        torch.cuda.set_rng_state(previous_state_gpu)

    base = UniformGaussian(ndim_cond, ind=[])  # Gaussian base distribution
    base = base.to(device)

    wrapper = RNVPFrEIAFlow(q0=base, freia_flow=inn)

    return wrapper


if __name__ == "__main__":
    ndim_in = 1
    ndim_cond = 1
    inn = create_conditional_RNVP_flow(ndim_in, ndim_cond, [128], 1, device="cpu")

    sample_data = torch.randn((10, ndim_in))
    sample_cond = torch.randn((10, ndim_cond))

    latent, forward_logdet = inn(sample_data, (sample_cond,))
    reconstruction, backward_logdet = inn(latent, (sample_cond,), rev=True)

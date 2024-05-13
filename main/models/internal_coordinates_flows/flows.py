# Code from normflows package (https://github.com/VincentStimper/normalizing-flows),
# adapted for conditional normalizing flows

import torch
import torch.nn as nn
from main.models.internal_coordinates_flows.couplings import (
    CircularCoupledRationalQuadraticSpline,
)
from main.models.flow_base import ConditionalFlowBase
from normflows.distributions.base import UniformGaussian


class ConditionalSplineFlow(ConditionalFlowBase):
    def __init__(self, q0: UniformGaussian, flows):
        """
        Normalizing Flow model to approximate target distribution using rational quadratic splines.
        Applies conditioning on CircularCoupledRationalQuadraticSpline flow layers.

        Args:
          q0: Base distribution
          flows: List of flows
        """

        super().__init__(q0)
        self.flows = nn.ModuleList(flows)

    def forward_and_log_det(self, z, context=None):
        """Transforms latent variable z to the flow variable x and
        computes log determinant of the Jacobian

        Args:
          z: Batch in the latent space
          context: Batch of conditional variables

        Returns:
          Batch in the space of the target distribution,
          log determinant of the Jacobian
        """

        log_det = torch.zeros(len(z), device=z.device)
        for flow in self.flows:
            if isinstance(flow, CircularCoupledRationalQuadraticSpline):
                z, log_d = flow(z, context=context)
            else:
                z, log_d = flow(z)
            log_det += log_d
        return z, log_det

    def inverse_and_log_det(self, x, context=None):
        """Transforms flow variable x to the latent variable z and
        computes log determinant of the Jacobian

        Args:
          x: Batch in the space of the target distribution
          context: Batch of conditional variables

        Returns:
          Batch in the latent space, log determinant of the
          Jacobian
        """

        log_det = torch.zeros(len(x), device=x.device)
        for i in range(len(self.flows) - 1, -1, -1):
            if isinstance(self.flows[i], CircularCoupledRationalQuadraticSpline):
                x, log_d = self.flows[i].inverse(x, context=context)
            else:
                x, log_d = self.flows[i].inverse(x)
            log_det += log_d
        return x, log_det

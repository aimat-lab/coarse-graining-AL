import torch.nn as nn
import torch
from normflows.distributions.base import UniformGaussian


class ConditionalFlowBase(nn.Module):
    def __init__(self, q0: UniformGaussian):
        """Base class for conditional normalizing flows

        Args:
            q0: Base distribution
        """

        super().__init__()
        self.q0 = q0

    def forward(self, z, context=None):
        """Transforms latent variable z to the variable x_int

        Args:
          z: Batch in the latent space
          context: Batch of conditional variables

        Returns:
          Batch in the space of the target distribution
        """

        x, log_jac_det = self.forward_and_log_det(z, context=context)
        return x

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

        raise NotImplementedError("forward_and_log_det not implemented")

    def inverse(self, x_int, context=None):
        """Transforms flow variable x_int to the latent variable z

        Args:
          x_int: Batch in the space of the target distribution
          context: Batch of conditional variables

        Returns:
          Batch in the latent space
        """

        z, log_jac_det = self.inverse_and_log_det(x_int, context=context)
        return z

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

        raise NotImplementedError("inverse_and_log_det not implemented")

    def sample(self, num_samples=1, context=None):
        """Sample from the target distribution

        Args:
          num_samples: Number of samples to generate
          context: Batch of conditional variables

        Returns:
          Batch of samples
          Batch of log probabilities
        """

        z, log_q = self.q0(num_samples)
        x_int, log_trafo = self.forward_and_log_det(z, context=context)

        return x_int, log_q - log_trafo

    def log_prob(self, x_int, context=None):
        """Computes log probability of a batch of samples

        Args:
          x_int: Batch of samples
          context: Batch of conditional variables

        Returns:
          Batch of log probabilities
        """

        z, log_trafo = self.inverse_and_log_det(x_int, context=context)
        log_q = self.q0.log_prob(z)

        return log_q + log_trafo

    def forward_kld(self, x_int, context=None):
        """Estimates forward KL divergence, see [arXiv 1912.02762](https://arxiv.org/abs/1912.02762)

        Args:
          x_int: Batch sampled from target distribution
          context: Batch of conditional variables

        Returns:
          Estimate of forward KL divergence averaged over batch
        """

        return -torch.mean(self.log_prob(x_int, context=context))

    def save(self, path):
        """Save state dict of model

        Args:
          path: Path including filename where to save model
        """
        torch.save(self.state_dict(), path)

    def load(self, path):
        """Load model from state dict

        Args:
          path: Path including filename where to load model from
        """
        self.load_state_dict(torch.load(path))

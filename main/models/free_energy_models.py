import torch
import torch.nn as nn
from typing import Tuple, List, Union
import functorch
import numpy as np
import wandb


class FCFreeEnergyModel(nn.Module):
    def __init__(
        self,
        size_of_latent_space: int = 1,
        hidden_layers: List[int] = [128],
        device: str = "cuda",
        seed: int = None,
        output_force: bool = True,
        use_2pi_periodic_representation: bool = False,
        standard_scaler: tuple = None,
        dropout: float = 0.0,
    ):
        """Get model to calculate the free energy based on the CG coordinates.

        Args:
            size_of_latent_space (int, optional): Size of latent space / CG space. Defaults to 1.
            hidden_layers (List[int], optional): Hidden layer dimensions. Defaults to [128].
            device (str, optional): Device. Defaults to "cuda".
            seed (int, optional): Seed. Defaults to None.
            output_force (bool, optional): Output forces (negative gradient of free energy)? Defaults to True.
            use_2pi_periodic_representation (bool, optional): Use 2pi-periodic pre-processing of the input? Defaults to False.
            standard_scaler (tuple, optional): Tuple of (mean, std) to use for standard scaling. Defaults to None.
            dropout (float, optional): Dropout probability. Defaults to 0.0.
        """

        super(FCFreeEnergyModel, self).__init__()

        self.size_of_latent_space = size_of_latent_space

        # To keep backwards compatibility, however, it would be much nicer to have
        # all layers in one ModuleList
        self.layers = nn.ModuleList()

        self.dropout = dropout
        if dropout > 0.0:
            self.dropout_layer = nn.Dropout(p=dropout)

        if not use_2pi_periodic_representation:
            previous = size_of_latent_space
        else:
            previous = 2 * size_of_latent_space

        for hidden_layer_dim in hidden_layers:
            self.layers.append(nn.Linear(previous, hidden_layer_dim, device=device))
            previous = hidden_layer_dim

        self.layers.append(nn.Linear(previous, 1, device=device))  # map to free energy

        self.initialize_weights(seed=seed)

        self.output_force = output_force
        self.use_2pi_periodic_representation = use_2pi_periodic_representation

        if standard_scaler is not None:
            self.scaler_m = standard_scaler[0]
            self.scaler_s = standard_scaler[1]
        else:
            self.scaler_m = None
            self.scaler_s = None

    def initialize_weights(self, seed: int = None):
        if seed is not None:
            previous_state_cpu = torch.random.get_rng_state()
            previous_state_gpu = torch.cuda.get_rng_state()

            torch.manual_seed(seed)

        weight_init = wandb.config.free_energy_ensemble.get("weight_init", "pytorch")
        if wandb.config.free_energy_ensemble.get("use_tf_init", False):  # legacy
            weight_init = "tf"

        if weight_init == "tf":
            # Use tf default initialization
            for layer in self.layers:
                torch.nn.init.xavier_uniform_(layer.weight)
                layer.bias.data.fill_(0.0)
        elif weight_init == "pytorch":
            # Use pytorch default initialization
            for layer in self.layers:
                layer.reset_parameters()
        elif weight_init == "gaussian_uniform":
            for layer in self.layers:
                layer.weight.data.normal_(0.0, np.random.uniform() * 2.9 + 0.1)
                layer.bias.data.normal_(0.0, np.random.uniform() * 2.9 + 0.1)
        else:
            raise ValueError(f"Unknown weight initialization: {weight_init}")

        if seed is not None:
            torch.random.set_rng_state(previous_state_cpu)
            torch.cuda.set_rng_state(previous_state_gpu)

    def forward(
        self, x: torch.Tensor
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        """Call module.

        Args:
            x (torch.Tensor): Input.

        Returns:
            Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]: Output of the model. Depending on self.output_force, either a tuple of (output, gradient) or just the output.
        """

        def compute_F(x):
            if self.use_2pi_periodic_representation:
                cos_rep = torch.cos(x)
                sin_rep = torch.sin(x)
                x = torch.cat([cos_rep, sin_rep], dim=-1)

            if self.scaler_m is not None and self.scaler_s is not None:
                x = (x - self.scaler_m) / self.scaler_s

            for i, layer in enumerate(self.layers):
                x = layer(x)

                if i < len(self.layers) - 1:
                    x = torch.sigmoid(x)

                    if self.dropout > 0.0:
                        x = self.dropout_layer(x)

            return x[..., 0], x[..., 0]  # works for both batched and not-batched data

        if self.output_force:
            # compute sample-wise gradients of the free energy:
            grad, output = functorch.vmap(functorch.grad(compute_F, has_aux=True))(x)

            F_force = -1 * grad

            return output, F_force

        else:
            return compute_F(x)[0]

import boltzgen as bg
import numpy as np
import torch
import wandb
import normflows as nf
from main.systems.aldp.coord_trafo import create_coord_trafo
from main.systems.aldp.coord_trafo import transform_trajectory
from torchinfo import summary
from main.models.internal_coordinates_flows.couplings import (
    CircularCoupledRationalQuadraticSpline,
)
from main.models.internal_coordinates_flows.flows import ConditionalSplineFlow


def create_RQS_flow(
    coordinate_trafo: bg.flows.CoordinateTransform,
    conditional=False,
    conditional_indices=[17, 44],  # phi, psi
    periodic_conditioning=True,
    use_fab_periodic_conditioning=True,
    use_cos_sin_periodic_representation_identity=True,
):
    # Make sure to adhere to the permutations
    ndim = 60

    ncarts = coordinate_trafo.transform.len_cart_inds
    permute_inv = coordinate_trafo.transform.permute_inv.cpu().numpy()
    dih_ind_ = coordinate_trafo.transform.ic_transform.dih_indices.cpu().numpy()
    std_dih = coordinate_trafo.transform.ic_transform.std_dih.cpu()

    ind = np.arange(ndim)
    ind = np.concatenate(
        [ind[: 3 * ncarts - 6], -np.ones(6, dtype=int), ind[3 * ncarts - 6 :]]
    )
    ind = ind[permute_inv]
    dih_ind = ind[dih_ind_]  # Dihedral indices in the final internal coordinate system

    ind_circ_dih = [
        0,
        1,
        2,
        3,
        4,
        5,
        8,
        9,
        10,
        13,
        15,
        16,
    ]  # Order in the final internal coordinate system is the same as in the original coordinate system

    ind_circ = dih_ind[ind_circ_dih]
    # print(sorted(ind_circ))
    bound_circ = np.pi / std_dih[ind_circ_dih]

    # Define the tail bounds of the rational quadratic splines (outside [-tail_bound, tail_bound] the identity is used)
    tail_bound = 5.0 * torch.ones(ndim)
    tail_bound[
        ind_circ
    ] = bound_circ  # 5 for non-circular coordinates, pi for circular coordinates

    circ_shift = wandb.config.flow_architecture["circ_shift"]

    # Flow layers
    layers = []
    n_layers = wandb.config.flow_architecture["n_blocks"]

    if conditional:
        ndim -= len(conditional_indices)

        # Remove conditional indices from ind_circ, shift the rest to the left
        offset = 0
        for conditional_index in sorted(conditional_indices):
            bound_circ = bound_circ[
                ind_circ != conditional_index - offset
            ]  # also modify this one and remove the respective entries
            ind_circ = ind_circ[ind_circ != conditional_index - offset]

            ind_circ[
                ind_circ > conditional_index
            ] -= 1  # TODO: Shouldn't this be "> conditional_index - offset"? Doesn't change anything for aldp though.
            offset += 1

        conditional_index = np.array(conditional_indices)
        tail_bound = np.delete(tail_bound, conditional_index)

    # Base distribution: Uniform for circular coordinates, Gaussian for the rest
    base_scale = torch.ones(ndim)
    base_scale[ind_circ] = (
        bound_circ * 2
    )  # 1 for non-circular coordinates, 2*pi for circular coordinates
    base = nf.distributions.UniformGaussian(
        ndim, ind_circ, scale=base_scale
    )  # [-pi, pi] uniform for circular coordinates, Gaussian with std 1 for the rest
    base.shape = (ndim,)

    for i in range(n_layers):
        bl = wandb.config.flow_architecture["blocks_per_layer"]
        hu = wandb.config.flow_architecture["hidden_units"]
        nb = wandb.config.flow_architecture["num_bins"]
        ii = wandb.config.flow_architecture["init_identity"]
        dropout = wandb.config.flow_architecture["dropout"]

        if i % 2 == 0:
            mask = nf.utils.masks.create_random_binary_mask(ndim, seed=i)
        else:
            mask = 1 - mask
        layers.append(
            CircularCoupledRationalQuadraticSpline(
                ndim,
                bl,
                hu,
                ind_circ,
                tail_bound=tail_bound,
                num_bins=nb,
                init_identity=ii,
                dropout_probability=dropout,
                mask=mask,
                num_cond_channels=len(conditional_indices) if conditional else 0,
                periodic_conditioning=periodic_conditioning,
                NO_frequencies=wandb.config.flow_architecture.get(
                    "periodic_conditioning_NO_frequencies", 1
                ),
                use_fab_periodic_conditioning=use_fab_periodic_conditioning,
                use_cos_sin_periodic_representation=use_cos_sin_periodic_representation_identity,
            )
        )

        mixing_type = wandb.config.flow_architecture["mixing_type"]

        if mixing_type == "affine":
            layers.append(nf.flows.InvertibleAffine(ndim, use_lu=True))
        elif mixing_type == "permute":
            layers.append(nf.flows.Permute(ndim))

        if wandb.config.flow_architecture["actnorm"]:
            layers.append(nf.flows.ActNorm(ndim))  # See Glow Paper

        if (
            i % 2 == 1 and i != n_layers - 1
        ):  # only after two blocks (with opposite masks) and not at the end
            if circ_shift == "constant":
                layers.append(
                    nf.flows.PeriodicShift(ind_circ, bound=bound_circ, shift=bound_circ)
                )
            elif circ_shift == "random":
                gen = torch.Generator().manual_seed(i)
                shift_scale = torch.rand([], generator=gen) + 0.5
                layers.append(
                    nf.flows.PeriodicShift(
                        ind_circ, bound=bound_circ, shift=shift_scale * bound_circ
                    )
                )

    # Map input to periodic interval
    layers.append(nf.flows.PeriodicWrap(ind_circ, bound_circ))  # Wrap to [-pi, pi]

    # normflows model
    if not conditional:
        flow = nf.NormalizingFlow(base, layers)
    else:
        flow = ConditionalSplineFlow(base, layers)

    summary(flow)

    return flow


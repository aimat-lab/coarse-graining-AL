import torch
import numpy as np
from scipy import integrate
import math
from typing import Union


def muller_potential(x, y):
    """Müller-Brown potential

    Args:
        x (torch.Tensor|np.ndarray): x coordinates
        y (torch.Tensor|np.ndarray): y coordinates

    Returns:
        torch.Tensor|np.ndarray: potential values

    Code adapted from https://cims.nyu.edu/~eve2/ztsMueller.m
    """

    aa = [-1, -1, -6.5, 0.7]
    bb = [0, 0, 11, 0.6]
    cc = [-10, -10, -6.5, 0.7]
    AA = [-200, -100, -170, 15]
    XX = [1, 0, -0.5, -1]
    YY = [0, 0.5, 1.5, 1]

    exp = torch.exp if isinstance(x, torch.Tensor) else np.exp

    value = 0
    for j in range(0, 4):
        value += AA[j] * exp(
            aa[j] * (x - XX[j]) ** 2
            + bb[j] * (x - XX[j]) * (y - YY[j])
            + cc[j] * (y - YY[j]) ** 2
        )
    return value


def muller_potential_regularized(x, y):
    """Müller-Brown potential, regularized similar to Noé et al. (2019)

    Args:
        x (torch.Tensor|np.ndarray): x coordinates
        y (torch.Tensor|np.ndarray): y coordinates

    Returns:
        torch.Tensor|np.ndarray: potential values

    Code adapted from https://cims.nyu.edu/~eve2/ztsMueller.m
    """

    E_high = -20.0
    E_max = 20.0
    Es = muller_potential(x, y)

    # Regularize the energies, similar to Noé et al. (2019):
    log = torch.log if isinstance(x, torch.Tensor) else np.log
    where = torch.where if isinstance(x, torch.Tensor) else np.where

    # Es[(E_high <= Es) & (Es < E_max)] = E_high + log(
    #    Es[(E_high <= Es) & (Es < E_max)] - E_high + 1
    # )
    # Es[Es >= E_max] = E_high + math.log(E_max - E_high + 1)

    # To make it compatible with functorch vmap:
    Es_regularized1 = E_high + log(Es - E_high + 1)
    Es_regularized2 = E_high + math.log(E_max - E_high + 1)

    # Combine the results using where:
    Es = where((E_high <= Es) & (Es < E_max), Es_regularized1, Es)
    Es = where(Es >= E_max, Es_regularized2, Es)

    return Es


def normalized_muller_brown_probability(
    x: Union[torch.Tensor, np.ndarray],
    y: Union[torch.Tensor, np.ndarray],
    Z: float,
    beta: float,
) -> Union[torch.Tensor, np.ndarray]:
    """Normalized Boltzmann probability for the muller brown potential.

    Args:
        x (torch.Tensor|np.ndarray): x coordinates
        y (torch.Tensor|np.ndarray): y coordinates
        Z (float): partition function (normalization constant)
        beta (float): 1/(kb*T)

    Returns:
        torch.Tensor: normalized probability values

    Code adapted from https://cims.nyu.edu/~eve2/ztsMueller.m
    """

    exp = torch.exp if isinstance(x, torch.Tensor) else np.exp

    return exp(-beta * muller_potential(x, y)) / Z


def get_muller_brown_Z(beta: float) -> float:
    """Compute the partition function of the Muller Brown potential.

    Args:
        beta (float): 1/(kb*T)
    """

    minx = -3.3
    maxx = 1.6
    miny = -1.2
    maxy = 3.2

    normalization_constant, _ = integrate.dblquad(
        lambda x, y: np.exp(-beta * muller_potential(x, y)),
        minx,
        maxx,
        lambda x: miny,
        lambda x: maxy,
    )

    return normalization_constant


def get_ground_truth_free_energies(
    s_values: np.ndarray, beta: float = 0.1
) -> np.ndarray:
    """Numerically compute the ground truth free energies along the s-coordinate for the Muller Brown potential.

    Args:
        s_values (np.ndarray): s values
        beta (float, optional): 1/(kb*T). Defaults to 0.1.

    Returns:
        np.ndarray: free energies in units of 1/beta
    """

    Z = get_muller_brown_Z(beta=beta)

    a = 1.0
    b = -1.0

    free_energies = np.zeros_like(s_values)

    for i, s in enumerate(s_values):
        integrand = lambda y: np.exp(
            -1.0 * beta * muller_potential(s / a - b / a * y, y)
        )
        result, _ = integrate.quad(integrand, -1, 3)

        free_energies[i] = -1.0 * np.log(result / Z)  # in units of 1/beta

    return free_energies

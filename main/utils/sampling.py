import numpy as np
from main.systems.mb.potential import muller_potential
import torch
from typing import Tuple
from typing import Callable
from typing import List
from typing import Union


def metropolis_monte_carlo_MB(
    steps: int, initial_state: np.ndarray, beta: float, step_size: float
) -> Tuple[np.ndarray, np.ndarray]:
    """Sample from the Muller-Brown potential using Metropolis Monte Carlo.

    Args:
        steps (int): number of steps to take.
        initial_state (np.ndarray): Initial coordinate.
        beta (float): 1/(kb*T).
        step_size (float): step size for MC.

    Returns:
        tuple: (samples, energies) where samples is a 2D ndarray of shape (steps+1, 2) and energies is a 1D ndarray of shape (steps+1,).
    """

    current_state = initial_state
    current_energy = muller_potential(*current_state)

    samples = np.empty((steps + 1, 2))
    samples[0] = current_state
    energies = np.empty(steps + 1)
    energies[0] = current_energy

    for i in range(1, steps + 1):
        proposed_state = current_state + step_size * np.random.randn(2)
        proposed_energy = muller_potential(*proposed_state)

        energy_diff = proposed_energy - current_energy
        acceptance_probability = np.minimum(1, np.exp(-beta * energy_diff))

        if np.random.rand() < acceptance_probability:
            current_state = proposed_state
            current_energy = proposed_energy

        samples[i] = current_state
        energies[i] = current_energy

    return samples, energies


def rejection_sampling(
    N: int,
    potential_function: Callable,
    min_potential_energy: float,
    bounds: List[Tuple[float, float]],
    beta: float = 0.1,
    use_torch_cuda: bool = False,
) -> Union[np.ndarray, torch.Tensor]:
    """Sample points from a potential function using rejection sampling.

    Args:
        N (int): number of points to sample.
        potential_function (callable): Potential energy function. It should accept a 2D ndarray,
            where each row represents a point in the space, and return a 1D ndarray of potential
            energies at these points.
        min_potential_energy (float): Estimate of the minimum potential energy.
        bounds (list of tuples): list of (min, max) for each dimension.
        beta (float, optional): 1/(kb*T). Defaults to 0.1.
        use_torch_cuda (bool, optional): Whether to use pytorch with cuda instead of numpy.ndarray.

    Returns:
        Union[np.ndarray, torch.Tensor]: 2D array of shape (N, D) where D is the dimension of the space.
    """

    D = len(bounds)  # dimension of the space
    i = 0

    # Initialize array to hold the samples
    samples = (
        np.zeros((N, D)) if not use_torch_cuda else torch.zeros((N, D), device="cuda")
    )

    exp = np.exp if not use_torch_cuda else torch.exp

    max_probability = exp(-1.0 * beta * min_potential_energy)

    while i < N:
        # draw random points from the uniform distribution for all dimensions
        points = (
            np.zeros((N - i, D))
            if not use_torch_cuda
            else torch.zeros((N - i, D), device="cuda")
        )
        for d in range(D):
            min_d, max_d = bounds[d]
            points[:, d] = (
                np.random.uniform(min_d, max_d, size=N - i)
                if not use_torch_cuda
                else (torch.rand(N - i, device="cuda") * (max_d - min_d) + min_d)
            )

        p = (
            np.random.uniform(0, 1, size=N - i)
            if not use_torch_cuda
            else torch.rand(N - i, device="cuda")
        )

        potential = potential_function(points)
        probability = exp(-1.0 * beta * potential)

        accept = p < probability / max_probability
        number_of_accepted = accept.sum()

        samples[i : i + number_of_accepted] = points[accept]
        i += number_of_accepted

    return samples

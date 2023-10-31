"""Scatter a particular neutrino off of many other "background" neutrinos.

Created 17 October 2023.

TODO: Fix scattering angle weights for 3D!
"""
import numpy as np
from numpy import pi, cos
from scipy.linalg import logm


def random_scatter(rho0, rho_background, n, *args, **kwargs):
    """Scatter a neutrino of interest off of background neutrinos `n` times."""
    theta = pi * np.random.rand(n)

    rho = np.array(rho0)
    yield rho

    for theta_ in theta:
        rho = scatter_background(rho, rho_background, *args, theta=theta_, **kwargs)
        yield rho


def scatter_background(rho, rho_background, *args, **kwargs):
    """Scatter a neutrino of interest with (2x2) density matrix `rho` off of a
    background neutrino with *independent* (2x2) density matrix
    `rho_background`, and return the new density matrix for the neutrino of
    interest."""
    rho_full = np.moveaxis(np.tensordot(rho, rho_background, axes=0), 1, 2)
    rho_full = scatter(rho_full, *args, **kwargs)
    rho = trace_out(rho_full)

    return rho


def scatter(rho, theta=pi/2, omega0_t=0.1):
    """Evolve the flavor density matrix of two neutrinos that scatter."""
    phase = np.exp(-2j * omega0_t * (1 - cos(theta)))

    # Time evolution matrix.
    N = 2
    N_states = 2
    U = np.zeros(2 * N * [N_states], dtype=complex)
    U[0, 0, 0, 0] = U[1, 1, 1, 1] = phase
    U[0, 1, 0, 1] = U[1, 0, 1, 0] = (phase + 1) / 2
    U[0, 1, 1, 0] = U[1, 0, 0, 1] = (phase - 1) / 2

    rho = matmul(matmul(U, rho), dagger(U))  # TODO
    return rho


def matmul(A, B):
    """Multiply two (2, 2, 2, 2) arrays as if they were (4, 4) matrices."""
    return np.einsum('ijkl,klmn', A, B)


def dagger(A):
    """Find the Hermitian conjugate of a (2, 2, 2, 2) array as if it was a
    (4, 4) matrix."""
    return np.moveaxis(A, (0, 1), (2, 3)).conjugate()


def trace_out(rho):
    """Take the trace with respect to the second neutrino."""
    return np.trace(rho, axis1=1, axis2=3)


def flavor_expval(rho):
    return rho[1, 1].real


def entropy(rho):
    return - np.trace(rho @ logm(rho)).real

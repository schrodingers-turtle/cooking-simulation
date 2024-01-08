"""Scatter neutrinos using density matrices.

Note: the functions expect states to have complex datatypes.

Created 17 October 2023.
"""
import numpy as np
from numpy import pi, sin, cos, arccos, identity
from scipy.linalg import logm


def random_scatter(initial_states, split_index, n, l, omega=(0, 0), theta=0.1, Theta='random', reproducible=False):
    """Scatter random pairs of particles `n` times and return all the states
    along the way. Scatter the particles with random angles and with vacuum
    oscillations.

    :param split_index: Where to split the states to apply two different vacuum
     oscillations.
    :param reproducible: Reset the random seed if `True`.
    :param Theta: Relative angle of (the momentum of) interacting neutrinos.
    :param theta: Neutrino mixing angle.
    :param omega: Vacuum oscillation frequency (in units of the coherent flavor
     conversion oscillation frequency mu).
    """
    if reproducible:
        np.random.seed(42)

    N = len(initial_states)
    particle1, particle2 = pick_random_pairs(N, n)

    if Theta == 'random':
        Theta = random_theta(n)
    elif Theta == 'randomMartin':
        absolute_directions = random_direction(N)
        cos_Theta = (absolute_directions[particle1] * absolute_directions[particle2]).sum(axis=1)
        Theta = arccos(cos_Theta)

        # For debugging:
        print(particle1)
        print(particle2)
        print(absolute_directions)
        print(cos_Theta)
        print(Theta)
    else:
        Theta = Theta * np.ones(n)

    states = np.array(initial_states)
    yield states

    for i, (p1, p2, Theta_) in enumerate(zip(particle1, particle2, Theta)):
        states = states.copy()
        states[[p1, p2]] = independent_scatter(*states[[p1, p2]], l=l, Theta=Theta_)

        for (omega_, states_) in zip(omega, (states[:split_index], states[split_index:])):
            if omega_:
                # Apply vacuum oscillations.
                states_[:] = propagate(states_, omega_ * 2/N * l, theta)

        yield states


def scatter_backgrounds(rho0, rho_background, n, *args, **kwargs):
    """Scatter a neutrino of interest off of background neutrinos `n` times."""
    Theta = random_theta(n)

    rho = np.array(rho0)
    yield rho

    for Theta_ in Theta:
        rho, _ = independent_scatter(rho, rho_background, *args, Theta=Theta_, **kwargs)
        yield rho


def independent_scatter(rho1, rho2, *args, **kwargs):
    """Scatter two independent neutrinos, then return each of their new
    independent density matrices."""
    rho_full = combine_rho(rho1, rho2)
    rho_full = scatter(rho_full, *args, **kwargs)
    rho1, rho2 = split_rho(rho_full)
    return rho1, rho2


def scatter(rho, Theta=pi/2, l=0.1):
    """Evolve the flavor density matrix of two neutrinos that scatter."""
    phase = np.exp(-2j * l * (1 - cos(Theta)))

    # Time evolution matrix.
    N = 2
    N_states = 2
    U = np.zeros(2 * N * [N_states], dtype=complex)
    U[0, 0, 0, 0] = U[1, 1, 1, 1] = phase
    U[0, 1, 0, 1] = U[1, 0, 1, 0] = (phase + 1) / 2
    U[0, 1, 1, 0] = U[1, 0, 0, 1] = (phase - 1) / 2

    rho = matmul(matmul(U, rho), dagger(U))  # TODO

    # This step is required in order to avoid floating point error that quickly
    # blows up (after ~50 scatters).
    trace = np.trace(np.trace(rho, axis1=0, axis2=2))
    rho /= trace

    return rho


def propagate(rho, omega_t, theta):
    """Propagate a neutrino in the vacuum, i.e., apply vacuum oscillations to
    `rho`.

    TODO: Check bugs from trace normalization.
     (Since the simulation matches the analytic analysis, this probably isn't a
     huge issue, but it could be nice to check it explicitly anyway.)"""
    delta = omega_t / 2
    sin_ = sin(2*theta)
    cos_ = cos(2*theta)
    U = cos(delta) * identity(2) + 1j * sin(delta) * np.array([[cos_, sin_], [sin_, -cos_]])
    rho = U @ rho @ U.T.conjugate()

    return rho


def combine_rho(rho1, rho2):
    """Combine `rho1` and `rho2` into `rho` via a tensor product."""
    return np.moveaxis(np.tensordot(rho1, rho2, axes=0), 1, 2)


def split_rho(rho):
    """Obtain each neutrino's density matrix by tracing out the other."""
    rho1 = np.trace(rho, axis1=1, axis2=3)
    rho2 = np.trace(rho, axis1=0, axis2=2)
    return rho1, rho2


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


def pick_random_pairs(N, shape):
    """Pick two random (different) integers in [0, `N`)."""
    # Any random choice.
    choice1 = np.random.randint(0, N, shape)

    # A random choice that's different from choice 1.
    choice2 = (choice1 + np.random.randint(1, N, shape)) % N

    return choice1, choice2


def random_theta(shape):
    """Return the `theta` angles of spherically uniform random 3D directions."""
    cos_theta = 2*np.random.rand(shape) - 1
    theta = arccos(cos_theta)

    return theta


def random_direction(shape):
    """Pick random 3D directions as unit vectors."""
    theta = random_theta(shape)
    phi = 2*pi * np.random.rand(shape)
    n_hat = np.moveaxis(np.stack([sin(theta)*cos(phi), sin(theta)*sin(phi), cos(theta)]), 0, -1)
    return n_hat

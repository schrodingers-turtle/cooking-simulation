"""Scatter neutrinos using density matrices.

Note: the functions expect states to have complex datatypes.

Created 17 October 2023.
"""
import numpy as np
from numpy import pi, sin, cos, exp, arccos, identity

pauli = np.array([
    [[0, 1], [1, 0]],
    [[0, -1j], [1j, 0]],
    [[1, 0], [0, -1]]
])


def random_scatter(initial_states, split_index, n, gamma, alpha1=(0, 0), alpha2=(0, 0), Theta='random',
                   reproducible=False):
    """Scatter random pairs of particles `n` times and return all the states
    along the way. Scatter the particles with random angles and with vacuum
    oscillations.

    :param split_index: Where to split the states to apply two different vacuum
     oscillations.
    :param reproducible: Reset the random seed if `True`.
    :param Theta: Relative angle of (the momentum of) interacting neutrinos.
    :param gamma: Coherent flavor oscillation strength: mu * Delta z.
    :param alpha1: Simultaneous vacuum oscillation strength: omega * Delta z.
    :param alpha2: Free vacuum oscillation strength:
     omega * (free, vacuum oscillation time per collision, for a single
     neutrino).
    """
    if reproducible:
        np.random.seed(42)

    N = len(initial_states)
    particleA, particleB = pick_random_pairs(N, n)

    alpha1_array = np.ones(N)
    alpha1_array[:split_index] = alpha1[0]
    alpha1_array[split_index:] = alpha1[1]

    if Theta == 'random':
        Theta = random_theta(n)
    elif Theta == 'randomMartin':
        absolute_directions = random_direction(N)
        cos_Theta = (absolute_directions[particleA] * absolute_directions[particleB]).sum(axis=1)
        Theta = arccos(cos_Theta)

        # For debugging:
        print(particleA)
        print(particleB)
        print(absolute_directions)
        print(cos_Theta)
        print(Theta)
    else:
        Theta = Theta * np.ones(n)

    states = np.array(initial_states)
    yield states

    for i, (pA, pB, Theta_) in enumerate(zip(particleA, particleB, Theta)):
        alpha1a, alpha1b = alpha1_array[[pA, pB]]

        states = states.copy()
        states[[pA, pB]] = independent_scatter(
            *states[[pA, pB]], gamma=gamma, alpha1a=alpha1a, alpha1b=alpha1b, Theta=Theta_
        )

        for (alpha2_, states_) in zip(alpha2, (states[:split_index], states[split_index:])):
            if alpha2_:
                # Apply vacuum oscillations.
                states_[:] = propagate(states_, alpha2_ * 2 / N)

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


def scatter(rho, Theta=pi/2, gamma=0.1, alpha1a=0.001, alpha1b=0.002):
    """Evolve the flavor density matrix of two neutrinos that scatter."""
    gammabar = gamma * (1 - cos(Theta))
    DeltaAlpha = alpha1b - alpha1a
    K = (DeltaAlpha ** 2 + 4 * gammabar ** 2) ** (1 / 2)

    # The eigenvector matrix.
    S = np.array([
        [(DeltaAlpha - K) / (2 * gammabar), (DeltaAlpha + K) / (2 * gammabar)],
        [1, 1]
    ])
    Sinv = np.array([
        [-gammabar / K, (1 + DeltaAlpha / K) / 2],
        [gammabar / K, (1 - DeltaAlpha / K) / 2]
    ])

    # The diagonalized part of e^-iHt.
    expJ = np.diag([exp(-1j * (-gammabar - K) / 2), exp(-1j * (-gammabar + K) / 2)])

    # The time evolution matrix.
    U = np.zeros((4, 4), dtype=complex)

    U[1:3, 1:3] = S @ expJ @ Sinv
    U[0, 0] = exp(-1j * (-alpha1a - alpha1b + gammabar) / 2)
    U[3, 3] = exp(-1j * (alpha1a + alpha1b + gammabar) / 2)

    # TODO: Maybe try `numpy.linalg.multi_dot` for speed
    #  (anywhere that it can be applied, like whenever two `@`'s are used).
    rho = U @ rho @ U.T.conjugate()

    # This step is required in order to avoid floating point error that quickly
    # blows up (after ~50 scatters).
    rho /= rho.trace()

    return rho


def propagate(rho, omega_t):
    """Propagate a neutrino in the vacuum, i.e., apply vacuum oscillations to
    `rho`, in the mass basis.

    TODO: Check bugs from trace normalization.
     (Since the simulation matches the analytic analysis, this probably isn't a
     huge issue, but it could be nice to check it explicitly anyway.
     It may not be an issue since `scatter` always renormalizes the trace every
     step, anyway.)
    """
    phase = exp(-1j * -omega_t/2)
    U = np.diag([phase, phase.conjugate()])
    rho = U @ rho @ U.T.conjugate()

    return rho


def combine_rho(rho1, rho2):
    """Combine `rho1` and `rho2` into `rho` via a tensor product."""
    return np.moveaxis(np.tensordot(rho1, rho2, axes=0), 1, 2).reshape(4, 4)


def split_rho(rho):
    """Obtain each neutrino's density matrix by tracing out the other."""
    rho = rho.reshape(2, 2, 2, 2)
    rho1 = np.trace(rho, axis1=1, axis2=3)
    rho2 = np.trace(rho, axis1=0, axis2=2)
    return rho1, rho2


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


def rho_from_bloch(bloch):
    """Convert Bloch vectors to 2x2 density matrices."""
    return (identity(2) + np.tensordot(bloch, pauli, axes=1)) / 2


def bloch_from_rho(rho):
    """Convert 2x2 density matrices to Bloch vectors."""
    return np.trace(np.dot(rho, pauli), axis1=-3, axis2=-1)

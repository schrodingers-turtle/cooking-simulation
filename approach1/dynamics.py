import numpy as np
from numpy import cos, pi


def random_scatter(psi0, n, random_angles=True, *args, **kwargs):
    """Scatter random pairs of particles `n` times and return the states along the way."""
    N = psi0.ndim
    particle1 = np.random.randint(0, N, n)                      # Any random particle.
    particle2 = (particle1 + np.random.randint(1, N, n)) % N    # A random particle that's different from particle 1.

    if random_angles:
        theta = pi * np.random.rand(n)
    else:
        theta = pi/2 * np.ones(n)

    psi = psi0
    yield psi

    for i, j, theta_ in zip(particle1, particle2, theta):
        psi = scatter_pair(psi, i, j, *args, theta=theta_, **kwargs)
        yield psi


def scatter_pair(psi, n, m, *args, **kwargs):
    """Scatter the particles `n` and `m` given the state `psi`, and return the new state."""
    N = psi.ndim
    index = psi_index(N, n, m)

    psi = psi.copy()
    psi[index[0]], psi[index[1]], psi[index[2]], psi[index[3]] = scatter(psi[index[0]], psi[index[1]], psi[index[2]], psi[index[3]], *args, **kwargs)

    return psi


def psi_index(N, n, m):
    """Return indices for the 4 configurations of particles n and m."""
    index = np.full((4, N), slice(None))
    index[:, n] = [0, 0, 1, 1]
    index[:, m] = [0, 1, 0, 1]

    index = [tuple(i) for i in index]

    return index


def scatter(ee, eu, ue, uu, theta=pi/2, omega0_t=1):
    """Scatter two neutrinos given their combined state components, and return the new components."""
    phase = np.exp(-2j * omega0_t * (1 - cos(theta)))
    ee = ee * phase
    uu = uu * phase

    c = (eu + ue) / 2
    d = (eu - ue) / 2
    eu = c*phase + d
    ue = c*phase - d

    return ee, eu, ue, uu


def flavor_eigenstate(flavors):
    """Make a flavor eigenstate with the given flavors for each neutrino."""
    N = len(flavors)

    psi = np.zeros(N * [2], dtype=complex)
    psi[tuple(flavors)] = 1

    return psi


if __name__ == '__main__':
    # Tests.

    psi = flavor_eigenstate([0, 1, 1, 1, 1, 1])
    print(psi)

    scatter_pair(psi, 0, 1, theta=pi/2, omega0_t=0.01)
    print(psi)

    scatter_pair(psi, 0, 2, theta=pi/2, omega0_t=0.01)
    print(psi)

    scatter_pair(psi, 0, 3, theta=pi/2, omega0_t=0.01)
    print(psi)

    scatter_pair(psi, 0, 4, theta=pi/2, omega0_t=0.01)
    print(psi)

    scatter_pair(psi, 0, 5, theta=pi/2, omega0_t=0.01)
    print(psi)

    print(np.sum(np.abs(psi[1, ...])**2))  # (Should equal 0.0005 as of now.)

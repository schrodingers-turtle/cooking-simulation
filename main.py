import numpy as np
from numpy import cos
from scipy.constants import pi


def scatter_pair(psi, n, m, *args, **kwargs):
    """Scatter the particles `n` and `m` given the state `psi`, and return the new state."""
    N = len(psi.shape)
    index = psi_index(N, n, m)

    # psi[index[0]], psi[index[1]], psi[index[2]], psi[index[3]] = scatter(psi[index[0]], psi[index[1]], psi[index[2]], psi[index[3]])
    psi[index] = scatter(*psi[index], *args, **kwargs)

    return psi


def psi_index(N, n, m):
    """Return indices for the 4 configurations of particles n and m."""
    # index = np.full((4, N), slice(None))
    # index[:, n] = [0, 0, 1, 1]
    # index[:, m] = [0, 1, 0, 1]
    #
    # index = [tuple(idx) for idx in index]

    index = N * [slice(None)]
    index[n] = [0, 0, 1, 1]
    index[m] = [0, 1, 0, 1]

    return tuple(index)


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


def psi_flavor_eigenstate(flavors):
    """Make a flavor eigenstate with the given flavors for each neutrino."""
    N = len(flavors)

    psi = np.zeros(N * [2], dtype=complex)
    psi[tuple(flavors)] = 1

    return psi


psi = psi_flavor_eigenstate([0, 1, 1])
print(psi)

scatter_pair(psi, 0, 1, theta=pi/2, omega0_t=0.05)
print(psi)

scatter_pair(psi, 0, 2, theta=pi/2, omega0_t=0.05)
print(psi)

print(np.sum(np.abs(psi[1, ...])**2))

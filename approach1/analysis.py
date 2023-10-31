import numpy as np
from scipy.linalg import logm


def density_matrix(psi, n):
    """The density matrix for the particle `n` from the multiparticle state `psi`."""
    N = psi.ndim

    psi_ = np.moveaxis(psi, n, 0)
    c_ab = psi_[:, np.newaxis]
    c_apb = psi_[np.newaxis, :]

    rho = (c_ab * c_apb.conj()).sum(axis=tuple(range(2, N + 1)))

    return rho


def entropy_rho(rho):
    """The Von-Neumann entropy of a density matrix."""
    return - np.trace(rho @ logm(rho))


def entropy(psi, n):
    return entropy_rho(density_matrix(psi, n))


def flavor_expval(psi, n):
    """The expectation value of the flavor of neutrino `n`. Electron flavor = 0, muon flavor = 1."""
    return np.sum(np.abs(psi.take(1, n)) ** 2)


def average_entropy(psi):
    return np.mean([entropy(psi, n) for n in range(psi.ndim)])


def average_flavor(psi):
    return np.mean([flavor_expval(psi, n) for n in range(psi.ndim)])

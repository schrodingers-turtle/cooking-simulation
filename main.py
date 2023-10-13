import numpy as np

from dynamics import flavor_eigenstate
from analysis import random_scatter, entropy, flavor_expval, average_entropy, average_flavor


def analyze(initial_flavors, n_scatters, *args, **kwargs):
    psi0 = flavor_eigenstate(initial_flavors)
    psis = random_scatter(psi0, n_scatters, *args, **kwargs)

    avg_entropy = np.array([average_entropy(psi) for psi in psis])
    avg_flavor = np.array([average_flavor(psi) for psi in psis])

    entropy0 = np.array([entropy(psi, 0) for psi in psis])
    flavor0 = np.array([flavor_expval(psi, 0) for psi in psis])

    return avg_entropy, avg_flavor, entropy0, flavor0


avg_entropy, avg_flavor, entropy0, flavor0 = analyze([1, 0, 0], n_scatters=100, omega0_t=0.1)
print(avg_entropy)

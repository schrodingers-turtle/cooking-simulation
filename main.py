import time

import numpy as np
from numpy import pi

from dynamics import flavor_eigenstate
from analysis import random_scatter, entropy, flavor_expval, average_entropy, average_flavor


def analyze(initial_flavors, n_scatters, verbose=True, *args, **kwargs):
    psi0 = flavor_eigenstate(initial_flavors)
    N = len(initial_flavors)

    entropies, flavors = [], []
    for i, psi in enumerate(random_scatter(psi0, n_scatters, *args, **kwargs)):
        if verbose:
            print(f"Scatter #{i}")

        entropies.append([entropy(psi, n) for n in range(N)])
        flavors.append([flavor_expval(psi, n) for n in range(N)])

    entropies, flavors = np.array(entropies), np.array(flavors)

    return entropies, flavors


psi0 = 10 * [1] + 10 * [0]
omega0_t = 0.1
n_scatters = round(20 * pi/omega0_t)

start = time.perf_counter()
entropies, flavors = analyze(psi0, n_scatters, omega0_t=0.1)
end = time.perf_counter()

print(entropies)
print(flavors)
print(entropies.shape, flavors.shape)

print(f"Elapsed time: {end - start}")

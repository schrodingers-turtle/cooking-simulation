"""Take a lot of data with different numbers of neutrinos."""
import pickle
import time

import numpy as np

from dynamics import flavor_eigenstate, random_scatter
from analysis import entropy, flavor_expval


def take_data(initial_flavors, n_scatters, verbose=True, *args, **kwargs):
    psi0 = flavor_eigenstate(initial_flavors)
    N = len(initial_flavors)

    if verbose:
        print(f"N: {N} ------------------------------------------")

    entropies, flavors = [], []
    for i, psi in enumerate(random_scatter(psi0, n_scatters, *args, **kwargs)):
        if verbose:
            print(f"Scatter #{i}")

        entropies.append([entropy(psi, n) for n in range(N)])
        flavors.append([flavor_expval(psi, n) for n in range(N)])

    entropies, flavors = np.array(entropies).T, np.array(flavors).T

    return entropies, flavors


start = time.perf_counter()

# Equal flavor distribution, N = 2 * (1 thru 11).
# data = [take_data(M*[0] + M*[1], n_scatters=1500, omega0_t=0.1) for M in range(1, 12)]

# 9 initial electron neutrinos, 7 initial muon neutrinos.
data = [take_data(9*[0] + 7*[1], n_scatters=3000, omega0_t=0.1)]

end = time.perf_counter()

with open('data3.pickle', 'wb') as file:
    pickle.dump(data, file)

print(data)
print(f"Time: {end - start}")

import numpy as np

from dynamics import scatter_pair, flavor_eigenstate


def random_scatter(psi, n, *args, **kwargs):
    """Scatter random pairs of particles `n` times and return the states along the way."""
    N = len(psi.shape)
    particle1 = np.random.randint(0, N, n)                      # Any random particle.
    particle2 = (particle1 + np.random.randint(1, N, n)) % N    # A random particle that's different from particle 1.

    psis = [psi]
    for i, j in zip(particle1, particle2):
        psis.append(scatter_pair(psis[-1], i, j, *args, **kwargs))

    return psis


psi = flavor_eigenstate([1, 0, 0])

psis = random_scatter(psi, 1000, omega0_t=0.1)

for psi in psis:
    print(np.sum(np.abs(psi[1, ...])**2))

psi = psis[-1]

print("---")
print(np.sum(np.abs(psi[1, ...])**2))
print(np.sum(np.abs(psi[0, ...])**2))

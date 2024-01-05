"""Take data without crashing your laptop.

Created 19 Nov 2023.
"""
import pickle

import numpy as np
from numpy import pi

from approach2.dynamics import random_scatter

accumulate = False

sample_period = 100

Ne = 50
Nx = 50
l = 0.01
n = 500000 * 8
omega = np.array([0.1, 0.2])
theta = 0.3
# Theta = 'random'
Theta = 'randomMartin'
# Theta = pi / 2

version = 2

filename = f'Ne-{Ne}-Nx-{Nx}-l-{l}-n-{n}-P-{sample_period}-v-{version}.pickle'

Pauli = np.array([
    [[0, 1], [1, 0]],
    [[0, -1j], [1j, 0]],
    [[1, 0], [0, -1]]
])

initial_states = np.array(Ne * [[[1, 0], [0, 0]]] + Nx * [[[0, 0], [0, 1]]], dtype=complex)

split_index = Ne
data = []

with open(filename, 'wb') as file:
    for i, states in enumerate(random_scatter(initial_states, split_index, n, l=l, omega=omega, Theta=Theta, theta=theta)):
        if i % sample_period == 0:
            if accumulate:
                Bloch = states[..., np.newaxis, :, :] @ Pauli
                Bloch = np.trace(Bloch, axis1=-1, axis2=-2).real
                Bloch_e = Bloch[:split_index].mean(axis=0)
                Bloch_x = Bloch[split_index:].mean(axis=0)
                Bloch_e_std = Bloch[:split_index].std(axis=0)
                Bloch_x_std = Bloch[split_index:].std(axis=0)

                data.append((Bloch_e, Bloch_x, Bloch_e_std, Bloch_x_std))
            else:
                data.append(states)
        print(f"Finished scatter #{i}")

    data = np.array(data)
    pickle.dump(data, file)

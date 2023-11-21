"""Take data without crashing your laptop.

Created 19 Nov 2023.
"""
import pickle

import numpy as np
from numpy import pi

from approach2.dynamics import random_scatter

accumulate = False

sample_period = 1000

M = 50
l = 0.001
n = 5000000 * 2
omega = np.array([0.1, 0.2])
theta = 0.1
# Theta = 'random'
Theta = pi / 2

filename = f'M-{M}-l-{l}-n-{n}-P-{sample_period}.pickle'

Pauli = np.array([
    [[0, 1], [1, 0]],
    [[0, -1j], [1j, 0]],
    [[1, 0], [0, -1]]
])

initial_states = np.array(M * [[[1, 0], [0, 0]]] + M * [[[0, 0], [0, 1]]], dtype=complex)

data = []

with open(filename, 'wb') as file:
    for i, states in enumerate(random_scatter(initial_states, n, l=l, omega=omega, Theta=Theta, theta=theta)):
        if i % sample_period == 0:
            if accumulate:
                Bloch = states[..., np.newaxis, :, :] @ Pauli
                Bloch = np.trace(Bloch, axis1=-1, axis2=-2).real
                Bloch_e = Bloch[:M].mean(axis=0)
                Bloch_x = Bloch[M:].mean(axis=0)
                Bloch_e_std = Bloch[:M].std(axis=0)
                Bloch_x_std = Bloch[M:].std(axis=0)

                data.append((Bloch_e, Bloch_x, Bloch_e_std, Bloch_x_std))
            else:
                data.append(states)
        print(f"Finished scatter #{i}")

    data = np.array(data)
    pickle.dump(data, file)

"""Take data without crashing your laptop.

Created 19 Nov 2023.
"""
import pickle

import numpy as np
from numpy import pi, sin, cos

from approach2.dynamics import random_scatter, rho_from_bloch

accumulate = False

sample_period = 100

Ne = 60
Nx = 40
gamma = 0.01
n = 500000
alpha1 = gamma * np.array([0.1, 0.2])
alpha2 = 0 * gamma * np.array([0.1, 0.2])
theta = 0.1
# Theta = 'random'
# Theta = 'randomMartin'
Theta = pi / 2

version = 5
reproducible = True

filename = f'Ne-{Ne}-Nx-{Nx}-l-{gamma}-n-{n}-P-{sample_period}-v-{version}.pickle'

pauli = np.array([
    [[0, 1], [1, 0]],
    [[0, -1j], [1j, 0]],
    [[1, 0], [0, -1]]
])

N = Ne + Nx
split_index = Ne

# initial_states = np.array(Ne * [[[1, 0], [0, 0]]] + Nx * [[[0, 0], [0, 1]]], dtype=complex)
initial_states = np.stack(
    Ne * [rho_from_bloch([sin(2 * theta), 0, cos(2 * theta)])]
    + Nx * [rho_from_bloch([-sin(2 * theta), 0, -cos(2 * theta)])],
    dtype=complex
)

data = []

with open(filename, 'wb') as file:
    for i, states in enumerate(
            random_scatter(initial_states, split_index, n, gamma=gamma, alpha1=alpha1, alpha2=alpha2, Theta=Theta,
                           reproducible=reproducible)):
        if i % sample_period == 0:
            if accumulate:
                Bloch = states[..., np.newaxis, :, :] @ pauli
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

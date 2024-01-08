"""Run the simulation and save data.

Note: Outdated!
"""
import pickle

import numpy as np

from approach2.dynamics import random_scatter


def main():
    M = np.array([10, 20, 50, 100, 200, 500, 1000])
    n = 10000 * np.ones_like(M)
    for M_, n_ in zip(M, n):
        print(f"Taking data for M = {M_}, n = {n_}.")
        take_data(M_, n_)


def take_data(M, n):
    """Run the simulation and save data to a file.

    Note: This is outdated and might not run correctly!
     Currently, this is not set up to use the new signature of `random_scatter`. Martin's random angles are not
     specified in the call to `random_scatter`.
    """
    filename = f"M-{M}-n-{n}.pickle"

    split_index = M
    states = random_scatter(initial_states(M), split_index, n)

    with open(filename, 'ab') as file:
        for states_ in states:
            pickle.dump(states_, file)


def initial_states(M):
    """`M` electron neutrinos and `M` muon neutrinos."""
    return np.array(M * [[[1, 0], [0, 0]]] + M * [[[0, 0], [0, 1]]], dtype=complex)


main()

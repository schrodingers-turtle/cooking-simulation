import pickle

import matplotlib.pyplot as plt
import numpy as np
from numpy.polynomial.polynomial import Polynomial


def t_cross(avg_e_flavors, N):
    """Return the time at which the average flavor of the initial electron
    neutrinos reaches 1/2, with an arbitrary constant factor, but that
    scales correctly with the number of neutrinos `N`. This crossing time is
    expected to be proportional to the equilibration time."""
    n_collisions = zero_crossings(avg_e_flavors - 1/2)[0][0]  # The first crossing.
    return n_collisions / N**2


def zero_crossings(array):
    """Find the indices where an array crosses from positive to negative.

    Note: Doesn't necessarily handle cases where array value(s) are exactly 0
    correctly.
    """
    return np.where(np.diff(array >= 0))


def avg_e_flavors(flavors):
    """Assumes the first half of the neutrinos are electron neutrinos, and the
    last half are muon neutrinos."""
    return flavors[:len(flavors)//2].mean(axis=0)


def plot():
    with open('../data.pickle', 'rb') as file:
        full_data = pickle.load(file)

    M = np.arange(len(full_data)) + 1
    N = 2*M
    t_cross_ = [t_cross(avg_e_flavors(flavors), N_) for (_, flavors), N_ in zip(full_data, N)]

    data = np.stack([N, t_cross_])

    # Cut off first data point, which is for only 2 neutrinos.
    data = data[:, 1:]

    data = np.log(data)

    plt.style.use('bmh')
    fig, ax = plt.subplots(figsize=(4, 3), dpi=200)

    ax.set_xlabel(r"$\ln(N)$")
    ax.set_ylabel(r"$\ln(t_{cross})$")

    color = 'black'
    ax.scatter(*data[:, :-5], marker='^', fc='none', ec=color, label="data")
    ax.scatter(*data[:, -5:], marker='s', fc='none', ec=color, label="data used in linear fit")
    # ax.scatter(N, t_cross_)

    xlim = np.array(ax.get_xlim())
    ylim = np.array(ax.get_ylim())
    fit = Polynomial.fit(*data[:, -5:], 1)
    line_endpoints = np.array([data[0, -5], xlim[1]])
    ax.plot(line_endpoints, fit(line_endpoints), ls=':', c='black', label="linear fit ($t_{cross} \propto N^{-0.5}$)")
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    print(f"Linear fit coefficients: {fit.convert().coef}")

    ax.legend(loc='lower left')

    fig.tight_layout()
    fig.savefig('trash.png')


plot()

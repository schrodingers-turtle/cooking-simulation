"""For tests of the analytic solution for approach 2 (for a neutrino of
interest in a background).

Created 11 Nov 2023.
"""
import numpy as np
from numpy import pi, cross, cos, arccos


def scatter_backgrounds(a0, b, n, mu_t):
    a0, b = np.array(a0), np.array(b)

    cos_theta = 2 * np.random.rand(n) - 1
    theta = arccos(cos_theta)

    a = [a0]
    for theta_ in theta:
        a.append(scatter(a[-1], b, mu_t, theta_))

    return np.array(a)


def scatter(a, b, mu_t, theta=pi/2):
    coeff = mu_t * (1 - cos(theta))
    da = - coeff * cross(a, b) - coeff**2 * (a - b)
    return a + da


a = scatter_backgrounds([0, 0, 1], [1, 0, 0], 1000, 0.1)
print(a)

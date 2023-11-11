import pickle

import matplotlib.pyplot as plt
from numpy.polynomial.polynomial import Polynomial

with open('n-10000/equil-n-collisions-vs-M.pickle', 'rb') as file:
    equil_n_collisions = pickle.load(file)

print(equil_n_collisions)

x = equil_n_collisions[:, 0]
y = equil_n_collisions[:, 1]
fit = Polynomial.fit(x, y, deg=1)

plt.plot(2*x, y, 's-', mfc='black')
plt.xlabel("$N$")
plt.ylabel("$n$ collisions")

plt.tight_layout()
plt.savefig('test.png')

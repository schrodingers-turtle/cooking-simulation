import os
import pickle

import numpy as np
from numpy.polynomial.polynomial import Polynomial

folder = 'n-10000'

equil_n_collisions = []
for filename in os.listdir(folder):
    if not filename.startswith('M-'):
        continue

    M = int(filename.split('-')[1])
    print(f"Doing M: {M}")

    average_disequilibrium = []
    with open(os.path.join(folder, filename), 'rb') as file:
        try:
            while True:
                states = pickle.load(file)
                average_disequilibrium.append(
                    np.mean([
                        np.abs(states[:M, 0, 0]).mean() - 0.5,
                        0.5 - np.abs(states[M:, 0, 0]).mean()
                    ])
                )
        except EOFError:
            pass

    x = np.arange(len(average_disequilibrium))
    y = np.log(average_disequilibrium)
    fit = Polynomial.fit(x, y, deg=1)

    equil_n_collisions.append((
        M,
        - 1 / fit.convert().coef[1]
    ))

equil_n_collisions.sort()
equil_n_collisions = np.array(equil_n_collisions)

save_path = os.path.join(folder, 'equil-n-collisions-vs-M.pickle')
with open(save_path, 'wb') as file:
    pickle.dump(equil_n_collisions, file)

import numpy as np
from itertools import product
from pathlib import Path
import pickle

Ls = [6, 8, 10, 12, 14, 16]
#ps = [0.025, 0.0375, 0.05, 0.0625, 0.075, 0.0875, 0.1, 0.125, 0.15, 0.2, 0.25, 0.3, 0.35, 0.45, 0.5]
ps = [0.025, 0.05, 0.075, 0.1, 0.125, 0.15, 0.175, 0.2, 0.225, 0.25, 0.275, 0.3, 0.325, 0.35, 0.375, 0.4]

paths = Path('./').glob('*.npy')
remaining = {key:5000 for key in product(Ls, ps)}

for path in paths:
    n_samples, L, p = str(path.stem).split(',')
    remaining[int(L), float(p)] = max(0, remaining[int(L), float(p)]-int(n_samples))

print(remaining, sep='\n')

pickle.dump(remaining, open('../remaining', 'wb'))

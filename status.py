import numpy as np
from pathlib import Path
import os

paths = list(Path('./processed_data/').glob('*.npy'))
for path in sorted(paths):
    arr = np.load(path)
    n_samples = arr.shape[0]
    print(str(path), ': ', arr.shape, sep='')
    if len(str(path.stem).split(','))==2:
        if int(str(path.stem).split(',')[0]) > 100:
            print(int(str(path.stem).split(',')[0]))
            os.remove(path)
        np.save('./processed_data/'+f'{n_samples},'+str(path.name), arr)
        os.remove(path)


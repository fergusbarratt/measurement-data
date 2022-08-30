from pathlib import Path
import numpy as np
data_loc = '../raw_data/'

paths = Path(data_loc).glob('*.npy')
data = {}
for path in paths:
    p, L, Q, *_ = str(path.stem).split(',')
    arr = np.load(path)
    print(arr.shape)
    if arr.shape == (0,):
        print(path.stem)
        raise Exception
    data[(int(L), float(p), int(Q))] = arr

data = dict(sorted(list(data.items())))
print(*[(key, value.shape) for key, value in data.items()], sep='\n')

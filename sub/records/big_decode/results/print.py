from pathlib import Path
import numpy as np
import os
data_loc = './'

paths = list(Path(data_loc).glob('*.npy'))

res = {}
for path in paths:
    N, L, p = str(path.stem).split(',')
    acc = np.load(path)
    assert (int(L), float(p)) not in acc
    res[(int(L), float(p))] = acc

results = sorted(res.items())
collated = {}
ps = set()
for key, value in results:
    if key[0] in collated:
        collated[key[0]] += [value]
    else:
        collated[key[0]] = [value]
    ps.add(key[1])

ps = list(sorted(ps))
Ls = list(collated.keys())
collated = {key:np.concatenate(np.expand_dims(value, 0), axis=0) for key, value in collated.items()}
print(collated, sep='\n')

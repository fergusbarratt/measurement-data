import numpy as np
from pathlib import Path
from sklearn.metrics import accuracy_score, log_loss
from scipy.stats import entropy

paths = list(Path('./data').glob('*.npy'))


data = {}
for path in paths:
    *_, L, p = str(path.stem).split(',')
    data[(int(L), float(p))] = np.load(str(path))

data = dict(sorted(data.items()))
print([x[1] for x in data if x[0] == 14])
print([x[1] for x in data if x[0] == 16])
print([x[1] for x in data if x[0] == 18])
print([x[1] for x in data if x[0] == 20])

to_del = []
for key in data:
    pass
#    if key[1] == 0.35:
#        to_del.append(key)
#    if key[1] == 0.5:
#        to_del.append(key)
for key in to_del:
    del data[key]

Lps = data.keys()
Ls = sorted(list(set([x[0] for x in Lps])))
ps = sorted(list(set([x[1] for x in Lps])))
collated = {}
for key, value in data.items():
    if key[0]  in collated:
        collated[key[0]].append(value[:, :19947, :])
    else:
        collated[key[0]] = [value[:, :19947, :]]

for key in collated:
    collated[key] = np.stack(collated[key])[:, :, :]

results = np.concatenate(np.stack(list(collated.values())).transpose([2, 0, 1, 3, 4]), axis=2)

np.save('processed_data/raw_data.npy', results)
np.save('processed_data/Ls.npy', np.array(Ls))
np.save('processed_data/ps.npy', np.array(ps))


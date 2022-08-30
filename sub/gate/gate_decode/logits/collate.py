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
#print([x[1] for x in data.keys() if x[0] == 8])
#print([x[1] for x in data.keys() if x[0] == 10])
print([x[1] for x in data.keys() if x[0] == 12])
print([x[1] for x in data.keys() if x[0] == 14])
#print([x[1] for x in data.keys() if x[0] == 16])
print([x[1] for x in data.keys() if x[0] == 18])
#print([x[1] for x in data.keys() if x[0] == 20])

to_del = []
for key in data:
    pass
#    if key[0] == 14:
#        to_del.append(key)
#    if key[0] == 12:
#        to_del.append(key)
#    if key[0] == 18:
#        to_del.append(key)
for key in to_del:
    del data[key]

Lps = data.keys()
Ls = sorted(list(set([x[0] for x in Lps])))
ps = sorted(list(set([x[1] for x in Lps])))
length = 20000
collated = {}
for key, value in data.items():
    if key[0]  in collated:
        collated[key[0]].append(value[:, :length, :])
    else:
        collated[key[0]] = [value[:, :length, :]]

for key in collated:
    collated[key] = np.stack(collated[key])[:, :, :]

print([x.shape for x in collated.values()])
print([x for x in collated])
results = np.concatenate(np.stack(list(collated.values())).transpose([2, 0, 1, 3, 4, 5]), axis=2).swapaxes(2, 3)
labels = np.concatenate([np.eye(2)[:, 0:1] for _ in range(length)]+[np.eye(2)[:, 1:2] for _ in range(length)], axis=1).T
print(results.shape, labels.shape)

np.save('processed_data/raw_data.npy', results)
np.save('processed_data/Ls.npy', np.array(Ls))
np.save('processed_data/ps.npy', np.array(ps))


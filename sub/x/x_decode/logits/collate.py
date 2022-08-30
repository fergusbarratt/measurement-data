import numpy as np
from pathlib import Path
from sklearn.metrics import accuracy_score, log_loss
from scipy.stats import entropy

paths = list(Path('./data').glob('*.npy'))


data = {}
labels = {}
for path in paths:
    if not str(path.stem).endswith('label'):
        *_, L, p = str(path.stem).split(',')
        data[(int(L), float(p))] = np.load(str(path))
    else:
        *_, L, p = str(path.stem).split(',')
        p, _ = p.split('_')
        labels[(int(L), float(p))] = np.load(str(path))

data = dict(sorted(data.items()))

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
label_collated = {}
for key, value in data.items():
    if key[0]  in collated:
        collated[key[0]].append(value)
        label_collated[key[0]].append(labels[key])
    else:
        collated[key[0]] = [value]
        label_collated[key[0]] = [labels[key]]

for key in collated:
    collated[key] = np.stack(collated[key])
    label_collated[key] = np.stack(label_collated[key])

print([x.shape for x in collated.values()])
print([x.shape for x in label_collated.values()])

size = np.max([x.shape[2] for x in list(collated.values())])
print(size)
results = [np.pad(a, ((0, 0), (0, 0), (0, (size-a.shape[2])))) for a in list(collated.values())]

results = np.stack(results)
labels = np.stack(list(label_collated.values()))
#labels = np.concatenate([np.eye(2)[:, 0:1] for _ in range(19947)]+[np.eye(2)[:, 1:2] for _ in range(19947)], axis=1).T
np.save('processed_data/raw_data.npy', results)
np.save('processed_data/labels.npy', labels)
np.save('processed_data/Ls.npy', np.array(Ls))
np.save('processed_data/ps.npy', np.array(ps))

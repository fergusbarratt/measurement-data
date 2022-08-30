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

print([x.shape for x in collated.values()])
results = np.concatenate(np.stack(list(collated.values())).transpose([2, 0, 1, 3, 4]), axis=2)
labels = np.concatenate([np.eye(2)[:, 0:1] for _ in range(19947)]+[np.eye(2)[:, 1:2] for _ in range(19947)], axis=1).T
print(results.shape, labels.shape)

accs = np.zeros(shape=results.shape[:2])
ces = np.zeros(shape=results.shape[:2])
mean_ents = np.zeros(shape=results.shape[:2])
for i, j in np.ndindex(*accs.shape):
    accs[i, j] = accuracy_score(np.argmax(labels, axis=1), np.argmax(results[i, j, :, :], axis=1))
    ces[i, j] = log_loss(labels[np.logical_not(np.isnan(results[i, j, :, :]))], results[i, j, :, :][np.logical_not(np.isnan(results[i, j, :, :]))]/np.expand_dims(np.sum(results[i, j, :, :][np.logical_not(np.isnan(results[i, j, :, :]))], axis=-1), axis=-1))
    #mean_ents[i,j] = np.mean(entropy(results[i, j, :, :]/np.expand_dims(np.sum(results[i, j, :, :], axis=-1), axis=-1), axis=1))
print(accs)
print(ces)
print(results.shape)
np.save('processed_data/accs.npy', accs)
np.save('processed_data/ces.npy', ces)
np.save('processed_data/raw_data.npy', results)
np.save('processed_data/Ls.npy', np.array(Ls))
np.save('processed_data/ps.npy', np.array(ps))


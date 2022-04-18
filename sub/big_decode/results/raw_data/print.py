from pathlib import Path
import numpy as np
import os
data_loc = '../../../processed_data/'


Ls = [6, 8, 10, 12, 14, 16, 18, 20]
ps = [0.05, 0.075, 0.1, 0.125, 0.15, 0.25, 0.35, 0.45, 0.5]

# read in the data
K = None
all_data = {}
for L in Ls:
    for p in ps:
        def end(L):
            return 2*L
            #return int(6*np.sqrt(L)+1)
        arr =  np.expand_dims(np.load(data_loc+f"{L},{p}.npy")[:K, :, :, :end(L)], 0) # drop last (fully measured) row
        all_data[(str(L), str(p))] = arr
        
# stack the data for different p
collated = {}
for key, value in sorted(all_data.items()):
    if key[0] in collated:
        mval = min(collated[key[0]].shape[1], value.shape[1])
        collated[key[0]] = np.concatenate([collated[key[0]][:, :mval], value[:, :mval]], axis=0)
    else:
        collated[key[0]] = value 

results = {}
sizes = {}
for key, value in collated.items():
    vals = []
    size = value.shape[1]
    for p_ind in range(len(ps)):
        vals.append(np.load(f'raw_data/decoded,{key},{p_ind}.npy'))
    results[int(key)] = np.concatenate([np.expand_dims(x, axis=0) for x in vals], axis=0)
    sizes[int(key)] = size

data = np.concatenate([np.expand_dims(x[1], 0) for x in sorted(results.items())], axis=0)
sizes = np.concatenate([np.expand_dims(x[1], 0) for x in sorted(sizes.items())], axis=0)
np.save('collated.npy', data)
np.save('sizes.npy', sizes)

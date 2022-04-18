from pathlib import Path
import shutil
import numpy as np
import os

collate = True
merge = False

if collate:
    paths = sorted(Path('data').glob('*.npy'))

    data = {}
    saved = {}

    batches = 0
    for path in paths:
        datum = np.load(path)
        _, L, p, _ = str(path.stem).split(',')
        print(int(L), float(p))
        L, p = int(L), float(p)
        if (L,p) in data:
            data[(L, p)].append(datum)
            if batches:
                if len(data[(L, p)]) == batches:
                    np.save(f'processed_data/{batches*56},{saved[(L, p)]},{L},{p}', np.concatenate(data[(L, p)], axis=0))
                    saved[(L, p)] += 1
                    data[(L, p)] = []
        else:
            saved[(L, p)] = 0
            data[(L, p)] = [datum]

    for key, value in data.items():
        (L, p) = key
        arr = np.concatenate(data[(L, p)], axis=0)
        np.save(f'processed_data/{arr.shape[0]},{L},{p}', arr)

if merge:
    paths = sorted(Path('../../processed_data/').glob('*.npy'))
    local_paths = sorted(Path('./processed_data/').glob('*.npy'))
    for path in local_paths:
        *_, old_L, old_p = str(path.stem).split(',')
        to_merge = []
        for other_path in paths:
            *_, L, p = str(other_path.stem).split(',')
            if L==old_L and p==old_p:
                to_merge += [other_path]
        if to_merge != []:
            merged_array = np.concatenate([np.load(path)]+[np.load(x) for x in to_merge], axis=0)
            save_path = '../../processed_data/'+str(merged_array.shape[0]) + ','+old_L+','+old_p+'.npy'
            print(path, '->', save_path)
            np.save(save_path, merged_array) # save the merged array
            for merged_path in to_merge: # remove anything that's been merged into save_path
                os.remove(path)
        else:
            array = np.load(path)
            save_path = '../../processed_data/'+str(array.shape[0]) + ','+old_L+','+old_p+'.npy'
            print(path, '->', save_path)
            np.save(save_path, array)

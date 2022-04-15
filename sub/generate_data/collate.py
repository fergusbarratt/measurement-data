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
    for path in paths:
        local_path = 'processed_data/' + str(path.name)
        if not os.path.exists(local_path): continue
        old_data = np.load(path)
        new_data = np.load(local_path)
        merged_data = np.concatenate([old_data, new_data], axis=0)
        print(old_data.shape, new_data.shape, merged_data.shape)
        np.save(path, merged_data)
        shutil.move(local_path, 'merged_data/'+path.name)
        assert not os.path.exists(local_path)
        
    paths = sorted(Path('processed_data/').glob('*.npy'))
    for path in paths:
        old_path = '../../processed_data/'+str(path.name)
        assert not os.path.exists(old_path)
        shutil.move(path, old_path)
        assert not os.path.exists(path)

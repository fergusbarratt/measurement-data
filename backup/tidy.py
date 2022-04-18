from pathlib import Path
import os
import shutil
import numpy as np
paths = sorted(Path('./').glob('*.npy'))
all_data = {}
for path in paths:
    *_, L, p = str(path.stem).split(',')

    if (int(L), float(p)) in all_data:
        all_data[(int(L), float(p))].append(str(path))
    else:
        all_data[(int(L), float(p))] = [str(path)]

print([x[1] for x in all_data.items()])
for key, paths in all_data.items():
    if len(paths)==1:
        data = np.load(paths[0])
        print(paths[0])
        if len(str(paths[0]).split(',')[0])>2:
            np.save(paths[0].replace(paths[0].split(',')[0], str(data.shape[0])), np.load(paths[0]))
            os.remove(paths[0])
        else:
            np.save(str(data.shape[0])+','+paths[0], np.load(paths[0]))
            os.remove(paths[0])
        print('\n')
    else:
        print('doubled')
        for path in paths:
            data = np.load(path)
            print(path)
            if len(str(path).split(',')[0])>2:
                np.save(path.replace(path.split(',')[0], str(data.shape[0])), np.load(path))
                os.remove(path)
            else:
                np.save(str(data.shape[0])+','+path, np.load(path))
                os.remove(path)
        print('\n')
        #arr = np.concatenate([np.load(path) for path in paths], axis=0)
        #np.save(str(arr.shape[0])+','+
        #if len(str(paths[0]).split(',')[0])>2:
        #    np.save(paths[0].replace(paths[0].split(',')[0], str(arr.shape[0])), arr)
        #else:
        #    print('...')
        #    np.save(str(data.shape[0])+','+paths[0], paths[0])
        #    os.remove(paths[0])
        #print(arr.shape)

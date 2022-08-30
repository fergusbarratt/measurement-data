'''Operates on output of read_data.py.
Stacks numpy arrays from folders, and store result from each 
folder as a numpy array (with the same name + .npy).
Also strips out corrupted npys (WHY do they get corrupted?)''' 
from pathlib import Path
import numpy as np
import os

folders = Path('./processed_data').glob('*/')
for folder in folders:
    fns = (x for x in Path(str(folder)).glob('*/') if x.is_dir())
    for fn in fns:
        if not os.path.exists(str(fn)+'.npy'):
            files = Path(str(fn)).glob('*.npy')
            name = str(fn)
            arrays = []
            for afile in files:
                try:
                    anarray = np.load(str(afile), allow_pickle=False)
                except Exception as y:
                    print(y)
                    continue
                arrays.append(np.squeeze(anarray))
            print(fn)
            print(np.array(arrays).shape)
            np.save(str(fn), np.array(arrays))

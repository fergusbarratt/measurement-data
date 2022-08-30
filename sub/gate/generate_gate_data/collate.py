from pathlib import Path
import shutil
import numpy as np
import os

collate = True
merge = False

if collate:
    paths = sorted(Path('data').glob('*.npz'))

    measurement_data = {}
    unitary_data = {}

    batches = 0
    for path in paths:
        measurement_datum = np.load(path)['records']
        unitary_datum = np.load(path)['unitaries']

        _, L, p, _ = str(path.stem).split(',')
        print(int(L), float(p))
        L, p = int(L), float(p)
        if (L,p) in measurement_data:
            measurement_data[(L, p)].append(measurement_datum)
            unitary_data[(L, p)].append(unitary_datum)
        else:
            measurement_data[(L, p)] = [measurement_datum]
            unitary_data[(L, p)] = [unitary_datum]

    for key in measurement_data.keys():
        (L, p) = key
        measurements = np.concatenate(measurement_data[(L, p)], axis=0)
        unitaries = np.concatenate(unitary_data[(L, p)], axis=0)
        pathname = f'processed_data/{L},{p},{measurements.shape[0]}'
        np.savez(pathname, measurements=measurements, unitaries=unitaries)

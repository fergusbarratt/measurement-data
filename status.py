import numpy as np
from pathlib import Path

paths = list(Path('./processed_data/').glob('*.npy'))
for path in sorted(paths):
    print(str(path), ': ', np.load(path).shape, sep='')


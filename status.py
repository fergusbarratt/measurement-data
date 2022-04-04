import numpy as np
from pathlib import Path

paths = list(Path('./').glob('*.npy'))
for path in sorted(paths):
    print(str(path), ': ', np.load(path).shape, sep='')


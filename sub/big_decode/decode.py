from functools import reduce
import argparse

from pathlib import Path
from joblib import Parallel, delayed
import joblib
import socket
import psutil
import time
from itertools import product
import os
from measurement_analysis.stat_mech import decode
import numpy as np
from tqdm import tqdm
data_loc = '../../processed_data/'

def get_host(a=None, b=None):
    core = psutil.Process().cpu_num()
    cpu_count = joblib.cpu_count()
    def f():
        time.sleep(5e-1)
        return psutil.Process().cpu_num()
    ret = sorted(list(Parallel(n_jobs=-1)(delayed(f)() for _ in range(56))))
    return socket.gethostname(), cpu_count, ret

def decode_records(records):
    q0s = Parallel(n_jobs=-1)(delayed(decode)(record) for record in tqdm(records[:, 0]))
    q1s = Parallel(n_jobs=-1)(delayed(decode)(record) for record in tqdm(records[:, 1]))
    a1 = 1-np.mean(q0s)
    a2 = np.mean(q1s)
    return (a1+a2)/2

if __name__ == "__main__":
    print('running')
    # Put the datasets on the host process
    #Ls = [6, 8, 10, 12, 14, 16, 18, 20]
    #ps = [0.05, 0.075, 0.1, 0.125, 0.15, 0.25, 0.35, 0.45, 0.5]
    #Ls = [6, 8, 10, 12, 14, 16, 18, 20]
    #ps = [0.025, 0.0375, 0.05, 0.0625, 0.075, 0.0875, 0.1, 0.125, 0.15, 0.2, 0.25, 0.35, 0.45, 0.5]

    paths = Path(data_loc).glob('*.npy')
    parser = argparse.ArgumentParser(description='..')
    parser.add_argument('L', type=int)
    parser.add_argument('p', type=float)
    args = parser.parse_args()
    data = {}
    records = []
    samples = []
    def end(L):
        return 3*int(np.sqrt(2*L))#2*L,
    for path in paths:
        n_samples, L, p = str(path.stem).split(',')
        if float(p) == args.p and int(L)==args.L:
            records.append(np.load(path)[:, :, :, :end(int(L))])
            samples.append(int(n_samples))

    if records == []:
        raise ValueError

    K = None
    results = [decode_records(x[:K]) for x in records]
    result = np.sum([x*y for x, y in zip(results, samples)])/np.sum(samples)
    
    others = list(Path('results/').glob(f'*{args.L},{args.p}.npy'))
    path = f'results/{np.sum(samples)},{args.L},{args.p}.npy'
    if not others:
        np.save(path, np.array(result))
    else:
        samples = [np.sum(samples)]
        results = [result]
        for path in others:
            n_samples, L, p = str(path.stem).split(',')
            samples.append(int(n_samples))
            results.append(np.load(path))
        result = np.sum([x*y for x, y in zip(results, samples)])/np.sum(samples)

    path = f'results/{np.sum(samples)},{args.L},{args.p}.npy'
    np.save(path, np.array(result))

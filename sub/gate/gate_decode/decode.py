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
data_loc = '../generate_gate_data/processed_data/'

def get_host(a=None, b=None):
    core = psutil.Process().cpu_num()
    cpu_count = joblib.cpu_count()
    def f():
        time.sleep(5e-1)
        return psutil.Process().cpu_num()
    ret = sorted(list(Parallel(n_jobs=-1)(delayed(f)() for _ in range(56))))
    return socket.gethostname(), cpu_count, ret

def decode_max(record):
    return np.argmax(decode(record))

def decode_logits(record, unitaries=None):
    unbiased_unitaries = 0*unitaries+np.ones(unitaries.shape)/2
    antibiased_unitaries = 1-unitaries
    unbiased_result = decode(record, unbiased_unitaries).real
    biased_result = decode(record, unitaries).real
    antibiased_result = decode(record, antibiased_unitaries).real
    result = np.stack([biased_result, antibiased_result, unbiased_result], axis=0)
    return result 

def get_logits(records, unitaries=None):
    q0s = np.array(Parallel(n_jobs=-1)(delayed(decode_logits)(record, unitary) for record, unitary in tqdm(list(zip(records[:, 0], unitaries[:, 0])))))
    q1s = np.array(Parallel(n_jobs=-1)(delayed(decode_logits)(record, unitary) for record, unitary in tqdm(list(zip(records[:, 1], unitaries[:, 0])))))
    q0s = q0s#/np.expand_dims(np.sum(q0s, axis=1), axis=-1)
    q1s = q1s#/np.expand_dims(np.sum(q1s, axis=1), axis=-1)
    return np.stack([q0s, q1s])

def decode_records(records):
    q0s = np.array(Parallel(n_jobs=-1)(delayed(decode_max)(record) for record in tqdm(records[:, 0])))
    q1s = np.array(Parallel(n_jobs=-1)(delayed(decode_max)(record) for record in tqdm(records[:, 1])))
    a1 = 1-np.mean(q0s)
    a2 = np.mean(q1s)
    return (a1+a2)/2

if __name__ == "__main__":
    print('running')
    paths = Path(data_loc).glob('*.npz')
    parser = argparse.ArgumentParser(description='..')
    parser.add_argument('L', type=int)
    parser.add_argument('p', type=float)
    args = parser.parse_args()
    data = {}
    records = []
    unitaries = []
    samples = []
    def end(L):
        return 2*L
    for path in paths:
        L, p, n_samples = str(path.stem).split(',')
        if float(p) == args.p and int(L)==args.L:
            f = np.load(path)
            records.append(f['measurements'][:, :, :, :end(int(L))])
            unitaries.append(f['unitaries'][:, :, :, :end(int(L))])
            samples.append(int(n_samples))

    path = f'logits/{np.sum(samples)},{args.L},{args.p}.npy'

    if records == []:
        raise ValueError
    if os.path.exists(path):
        raise ValueError

    K = None
    f = 0
    results = [get_logits(x[f:K], y[f:K]) for x, y in zip(records, unitaries)]
    result = np.concatenate([x for x in results], axis=1)
    path = f'logits/{np.sum(samples)},{args.L},{args.p}.npy'
    np.save(path, np.array(result))

import quimb.tensor as qtn
import random
import quimb as qu
import os
import gc
import numpy as np
import itertools
import matplotlib.pyplot as plt
import time 
from functools import reduce
from tqdm import tqdm

from tebdm import TEBDm

import scipy.stats as stats

import pickle
from pathlib import Path
from joblib import Parallel, delayed

plt.style.use('seaborn-whitegrid')
Z = np.array([[1, 0], [0, -1]])
Sp, Sm = np.array([[0, 1], [0, 0]]), np.array([[0, 0], [1, 0]])

def get_correlations(final_state, op1=Z, op2=Z):
    L = final_state._L
    summer = qtn.tensor_gen.MPS_product_state([np.array([1.0, 1.0])] * L)
    corr_matrix = np.ones((L, L))
    z_vec = np.zeros(L)
    ZZ = np.kron(op1, op2)
    def zz(i, j):
        return (
            1 / 2
            if i == j
            else 0
            if i < j
            else final_state.gate(ZZ, (i, j)).H @ summer
        )

    corr_matrix = np.array([[zz(i, j) for i in range(L)] for j in range(L)])
    corr_matrix = corr_matrix + corr_matrix.T

    batch = 1  # Get the local operators in batches. i.e. < Z_1+Z_2+Z_3 ...> + <Z_5+Z_6+...>
    op = reduce(
        lambda x, y: x @ y,
        [
            reduce(np.kron, [np.eye(2)] * k + [op1] + [np.eye(2)] * (batch - k - 1))
            for k in range(batch)
        ],
    )

    def z(i):
        return final_state.gate(op, list(range(i, i + batch))).H @ summer

    z_vec = np.array([z(i) for i in range(0, L, batch)])
    z_vec = np.expand_dims(z_vec, 0)
    mat = np.concatenate([corr_matrix, z_vec], axis=0)  # variance
    return mat

def get_strings(final_state, op=Z):
    L = final_state._L
    summer = qtn.tensor_gen.MPS_product_state([np.array([1.0, 1.0])] * L)
    def zz(i, j):
        if i == j: 
            return 1/2
        if i < j:
            return 0
        else:
            state = final_state.copy()
            for k in range(i-j):
                state.gate_(op, j+k)
            return state.H @ summer

    corr_matrix = np.array([[zz(i, j) for i in range(L)] for j in range(L)])
    corr_matrix = corr_matrix + corr_matrix.T

    return corr_matrix

def get_quantum_correlations(final_state, op1=Sp, op2=Sm):
    L = final_state._L
    Z = np.array([[1, 0], [0, -1]])
    corr_matrix = np.ones((L, L))
    z_vec = np.zeros(L)
    ZZ = np.kron(op1, op2)
    qnorm = final_state.H @ final_state
    def zz(i, j):
        return (
            1 / 2
            if i == j
            else 0
            if i < j
            else final_state.gate(ZZ, (i, j)).H @ final_state / qnorm
        )

    corr_matrix = np.array([[zz(i, j) for i in range(L)] for j in range(L)])
    corr_matrix = corr_matrix + corr_matrix.T

    batch = 1  # Get the local operators in batches. i.e. < Z_1+Z_2+Z_3 ...> + <Z_5+Z_6+...>
    op = reduce(
        lambda x, y: x @ y,
        [
            reduce(np.kron, [np.eye(2)] * k + [op1] + [np.eye(2)] * (batch - k - 1))
            for k in range(batch)
        ],
    )

    def z(i):
        return final_state.gate(op, list(range(i, i + batch))).H @ final_state / qnorm

    z_vec = np.array([z(i) for i in range(0, L, batch)])
    z_vec = np.expand_dims(z_vec, 0)
    mat = np.concatenate([corr_matrix, z_vec], axis=0)  # variance
    return mat

def get_quantum_correlations_unnormalised(final_state, op1=Sp, op2=Sm):
    L = final_state._L
    Z = np.array([[1, 0], [0, -1]])
    corr_matrix = np.ones((L, L))
    sp_vec = np.zeros(L)
    SpSm = np.kron(op1, op2)
    qnorm = 1#final_state.H @ final_state
    def spsm(i, j):
        return (
            1 / 2
            if i == j
            else 0
            if i < j
            else final_state.gate(SpSm, (i, j)).H @ final_state / qnorm
        )

    corr_matrix = np.array([[spsm(i, j) for i in range(L)] for j in range(L)])
    corr_matrix = corr_matrix + corr_matrix.T

    batch = 1  # Get the local operators in batches. i.e. < Z_1+Z_2+Z_3 ...> + <Z_5+Z_6+...>
    op = reduce(
        lambda x, y: x @ y,
        [
            reduce(np.kron, [np.eye(2)] * k + [op1] + [np.eye(2)] * (batch - k - 1))
            for k in range(batch)
        ],
    )

    def sp(i):
        return final_state.gate(op, list(range(i, i + batch))).H @ final_state / qnorm

    sp_vec = np.array([sp(i) for i in range(0, L, batch)])
    sp_vec = np.expand_dims(sp_vec, 0)
    mat = np.concatenate([corr_matrix, sp_vec], axis=0)  # variance
    return mat

def get_measurement_record(final_state):
    L = final_state._L
    summer = qtn.tensor_gen.MPS_product_state([np.array([1.0, 1.0])] * L)

    all_ps = np.array([max(min(np.real((1+final_state.gate(Z, i).H@summer)/2), 1.), 0.) for i in range(L)])
    all_ms = np.array([np.random.choice([-1, 1], p=[1-p, p]) for p in all_ps])
    rec = np.concatenate([final_state.measurement_record, all_ms.reshape(len(all_ms), 1)], axis=-1)
    return rec

def get_norm(final_state):
    return final_state.norm

def get_norm2(final_state):
    return final_state.H @ final_state

def load_MPS_from_dir(data_dir, fns=[lambda x: x], N=None, par='joblib', process_prefix='processed_'):
    """
    data_dir: Dir to read the mps files from. Will use pickle, and assume they have the prefix .dmp.
    fns: list of functions to evaluate on the mps. 
    N: number of files per key to evaluate on. 
    par: ['dask', 'joblib'] whether to parallelise using dask or joblib.
    process_prefix: what to append to data_dir. Writes the outputs here. 
    """
    processed_data_dir = process_prefix + data_dir
    ret = {}
    pathlist = list(Path(data_dir).glob('*.dmp'))
    for path in tqdm(pathlist):
        opts = path.stem.split(',')
        p, L, Q, T = float(opts[0]), *[int(x) for x in opts[1:-1]]
        if (p, L, Q, T) in ret:
            ret[(p, L, Q, T)].append(str(path))
        else:
            ret[(p, L, Q, T)] = [str(path)]

    all_paths = list(ret.items())
    random.shuffle(all_paths)
    for key, paths in tqdm(all_paths):
        dir_name = processed_data_dir + '/'+f'{key}'[1:-1].replace(" ", "")+',' + reduce(lambda x, y: x+','+y, [fn.__name__ for fn in fns])
        os.makedirs(dir_name, exist_ok=True)
        def mapped_fn(path):
            qid = path.split(',')[-1].split('.')[0]
            filename = dir_name+'/'+'file'+qid
            if not os.path.exists(filename+'.npy'):
                # if the path we're gonna store this at already exists, don't recompute it. 
                Path(filename+'.npy').touch() # create it, so that other workers can tell if it's being worked on. 
                assert os.path.exists(filename+'.npy')
                try:
                    net = qu.load_from_disk(str(path))
                except Exception as y:
                    print(y)
                out = np.array([fn(net) for fn in fns])
                np.save(filename, out)
                assert os.path.exists(filename+'.npy')
                del net
                gc.collect()
                return None
            else:
                return None

        if par=='joblib':
            Parallel(n_jobs=-1)(delayed(mapped_fn)(path) for path in tqdm(paths[:N], leave=False)) # batch for all L
        elif par==None:
            for path in tqdm(paths[:N], leave=False):
                mapped_fn(path)
        else:
            raise Exception
        
    return ret

if __name__=='__main__':
    timestamp = 'records'
    N = None
    data_dir = 'data/data_'+str(timestamp)

    par = 'joblib'
    local = True
    fns = [get_measurement_record]
    load_MPS_from_dir(data_dir, fns=fns, N=N, par=par)

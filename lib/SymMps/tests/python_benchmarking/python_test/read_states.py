import quimb.tensor as qtn
import random
import quimb as qu
import sys
import os
import gc
import numpy as np
import matplotlib.pyplot as plt
from functools import reduce
from tqdm import tqdm

from tebdm import TEBDm

import dask

import pickle
from pathlib import Path
from joblib import Parallel, delayed

plt.style.use('seaborn-whitegrid')
Z = np.array([[1, 0], [0, -1]])
Sp, Sm = np.array([[0, 1], [0, 0]]), np.array([[0, 0], [1, 0]])

def get_correlations(final_state, op1=Z, op2=Z):
    L = len(final_state.arrays)
    ZZ = np.kron(op1, op2)
    def zz(i, j):
        return (
            1 / 2
            if i == j
            else 0
            if i < j
            else final_state.gate(ZZ, (i, j)).H @ final_state
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
        return final_state.gate(op, list(range(i, i + batch))).H @ final_state

    z_vec = np.array([z(i) for i in range(0, L, batch)])
    z_vec = np.expand_dims(z_vec, 0)
    mat = np.concatenate([corr_matrix, z_vec], axis=0)  # variance
    return mat

def get_strings(final_state, op=Z):
    L = len(final_state.arrays)
    def zz(i, j):
        if i == j: 
            return 1/2
        if i < j:
            return 0
        else:
            temp_state = final_state.copy()
            for k in range(i-j):
                temp_state.gate_(op, j+k)
            return temp_state.H @ final_state

    corr_matrix = np.array([[zz(i, j) for i in range(L)] for j in range(L)])
    corr_matrix = corr_matrix + corr_matrix.T

    return corr_matrix

def get_temporal_entanglement(final_state):
    return final_state.es

def get_spatial_entanglement(final_state):
    L = len(final_state.arrays)
    es = np.array([final_state.entropy(i) for i in range(1, L)])
    return es

def get_norm(final_state):
    return final_state.norm

def get_norm2(final_state):
    return final_state.H @ final_state

def load_MPS_from_dir(data_dir, fns=[lambda x: x], N=None, par='dask', process_prefix='processed_'):
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
    for path in pathlist:
        opts = path.stem.split(',')
        p_u, p_m, L, T = float(opts[0]), float(opts[1]), *[int(x) for x in opts[2:-1]]
        if (p_u, p_m, L, T) in ret:
            ret[(p_u, p_m, L, T)].append(str(path))
        else:
            ret[(p_u, p_m, L, T)] = [str(path)]

    all_paths = list(ret.items())
    random.shuffle(all_paths)
    for key, paths in all_paths:
        dir_name = processed_data_dir + '/'+f'{key}'[1:-1].replace(" ", "")+',' + reduce(lambda x, y: x+','+y, [fn.__name__ for fn in fns])
        os.makedirs(dir_name, exist_ok=True)
        def mapped_fn(path):
            qid = path.split(',')[-1].split('.')[0]
            filename = dir_name+'/'+'file'+qid
            if not os.path.exists(filename+'.npy'):
                # if the path we're gonna store this at already exists, don't recompute it. 
                Path(filename+'.npy').touch() # create it, so that other workers can tell if it's being worked on. 
                assert os.path.exists(filename+'.npy')
                net = qu.load_from_disk(str(path))
                out = np.array([fn(net) for fn in fns])
                np.save(filename, out, allow_pickle=False)
                assert os.path.exists(filename+'.npy')
                del net
                gc.collect()
                return None
            else:
                return None

        if par=='joblib':
            Parallel(n_jobs=-1)(delayed(mapped_fn)(path) for path in tqdm(paths[:N])) # batch for all L
        elif par=='dask':
            for path in paths[:N]:
                ret[key].append(dask.delayed(mapped_fn)(path))
        elif par==None:
            for path in tqdm(paths[:N], leave=False):
                mapped_fn(path)
        else:
            raise Exception
        
    if par=='dask':
        ret = dask.compute(ret, traverse=True)
    return ret

if __name__=='__main__':
    #paths = [str(x) for x in Path('data').glob('*/') if str(x.stem).startswith('data')]
    paths = ['data']
    for data_dir in paths:
        print(data_dir)
        N = None
        data_dir = data_dir

        par = 'joblib'
        local = True
        fns = [get_correlations]

        load_MPS_from_dir(data_dir, fns=fns, N=N, par=par)

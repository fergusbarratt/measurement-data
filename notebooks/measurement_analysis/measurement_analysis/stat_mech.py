import quimb.tensor as qtn
import quimb as qu
import pickle
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path
import networkx as nx
from functools import reduce

from .tebdm import TEBDm
import gc

import joblib
plt.style.use('seaborn-whitegrid')

def state(N, Q) :
    psi = np.array([int(bin(n).count("1")==Q) for n in range(2**N)])
    return qu.qu(psi/np.sum(psi))

def evolve_state(p, L, T, Q, record):
    Q = L//2-Q
    summer = qtn.tensor_gen.MPS_product_state([np.array([1.0, 1.0])] * L)
    #psi0 = summer.copy(deep=True) / (summer.H @ summer)  # start from all + state

    tebd = TEBDm(p, psi0, H=np.ones((4, 4)))

    tebd.split_opts["cutoff"] = 1e-8 
    be_b = []

    #print(record)
    try:
        for psit in tebd.at_times(np.arange(T), measurement_locations=record, progbar=False, tol=1e-3):    
            #be_b += [psit.entropy(L//2)]    
            pass
    except Exception:
        # no trajectories are compatible with this charge value. 
        return 0
    
    final_state = tebd.pt
    final_state.entanglement_history = np.array(be_b)
    final_state.norm = tebd.norm
    return np.prod(final_state.norm)

def decode(record):
    L, T = record.shape
    T = (T-1)//2
    return evolve_state(0, L, T, 0, record.T) < evolve_state(0, L, T, 1, record.T)

if __name__ == '__main__':
    Ls = [18]
    T = lambda L: 2*L
    data_dir = f'data'
    dirpaths = sorted(list(Path('../../processed_data/').glob('*.npy')))

    K = 100
    results = {}
    for dirpath in dirpaths[:1]:
        for Q in [0, 1]:
            arecord = np.load(dirpath)
            N_samples, _, L, T = arecord.shape
            p = str(dirpath.stem).split(',')[-1]
            T = (T-1)//2

            records = np.load(dirpath)[:K, Q, :, :-1]
            mats0 = joblib.Parallel(n_jobs=-1)(joblib.delayed(evolve_state)(0, L, T, 0, record.T, data_dir) for record in tqdm(records))
            mats1 = joblib.Parallel(n_jobs=-1)(joblib.delayed(evolve_state)(0, L, T, 1, record.T, data_dir) for record in tqdm(records))
            acc = np.mean(np.array(mats1)>np.array(mats0))
            results[(Q, L, p)] = acc


    pickle.dump(results, open('../../results', 'wb'))
    print(results)
    print('done, moving folder')

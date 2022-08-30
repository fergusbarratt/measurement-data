import quimb.tensor as qtn
import quimb as qu
import os
import numpy as np
import random
import uuid
import matplotlib.pyplot as plt
from tqdm import tqdm
from tebdm import TEBDm
from measurement_analysis.stat_mech import decode
import gc
import joblib
from functools import lru_cache

plt.style.use('seaborn-whitegrid')

def state(N, Q):
    psi = np.array([int(bin(n).count("1")==Q) for n in range(2**N)])
    return qu.qu(psi/np.sum(psi))

def mps_state(N, Q):
    astate =  state(N, Q)
    return qtn.MatrixProductState.from_dense(astate, [2]*N)

np.random.seed(501)
def evolve_state(p, L, T, Qs, n, data_dir):
    logits = []
    for Q in Qs:
        summer = qtn.tensor_gen.MPS_product_state([np.array([1.0, 1.0])] * L)

        astate =  state(L, L//2-Q)
        psi0 =  qtn.MatrixProductState.from_dense(astate, [2]*L)
        psi0 = psi0 #/ (psi0.H @ summer)  # start from all + state

        tebd = TEBDm(p, psi0, H=np.ones((4, 4)), imag=True)

        tebd.split_opts["cutoff"] = 1e-16
        be_b = []

        for psit in tebd.at_times(np.arange(1, T+1), progbar=False, tol=1e-3):    
            pass
        
        final_state = tebd.pt

        record = tebd.measurement_locations
        logits.append(decode(record))
    return np.array(logits)

if __name__ == '__main__':
    Ls = [8, 10, 12, 14]
    ps = [0.025, 0.05 , 0.075, 0.1, 0.125, 0.15 , 0.175, 0.2, 0.25, 0.3]
    np.save('data/Ls', np.array(Ls))
    np.save('data/ps', np.array(ps))

    Qs = [0, 1]
    N = 5000
    H = np.zeros((4, 4))
    ϵ = 1e-10
    T = lambda L: L
    timestamp = 'records'#time.time()
    data_dir = f'data/data_{timestamp}'
    os.makedirs(data_dir, exist_ok=True)

    all_logits = []
    for L in Ls:
        p_logits = []
        for p in ps:
            print('calculating')
            logits = joblib.Parallel(n_jobs=-1)(joblib.delayed(evolve_state)(p, L, T(L), Qs, n, data_dir) for n in tqdm(range(N)))
            p_logits.append(np.array(logits))
        all_logits.append(np.array(p_logits))
    np.save(f'data/raw_data.npy', np.array(all_logits))
    print('done')

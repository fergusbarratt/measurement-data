import numpy as np
import os
import quimb.tensor as qtn
import quimb as qu
from tebdm import TEBDm
from joblib import Parallel, delayed
from tqdm import tqdm
import gc
I = np.array([[1, 0], [0, 1]])
H = np.array([[1, 1], [-1, 1]])/np.sqrt(2)
CNOT = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])

def evolve_state(p_u, p_m, L, T, n, data_dir):
    #psi0 = qtn.tensor_gen.MPS_product_state([np.array([1.0, 1.0])/np.sqrt(2)]*L)
    psi0 = qtn.tensor_gen.MPS_neel_state(L, down_first=False)
    U = CNOT@np.kron(H, I)
    for i in range(0, L, 2):
        psi0.gate_split(U, (i, i+1), inplace=True) # think this is tiled singlets

    psi0.right_compress()

    es = []
    tebd = TEBDm(p_u, p_m, psi0, np.zeros((4, 4)), imag=True)
    tebd.split_opts["cutoff"] = 1e-8
    for state in tebd.at_times(np.arange(T), dt=1, progbar=False):
        es.append(state.entropy(L//2))
    state.es = np.array(es)

    qid = qtn.tensor_core.rand_uuid() 
    qu.save_to_disk(state, data_dir + f'/{p_u},{p_m},{L},{T},{qid}.dmp')

    del state
    del psi0
    del es
    gc.collect()

if __name__ == '__main__':
    Ls = [8]
    T = lambda L: 4*L
    N = 1000
    #p_ms = np.linspace(0.2, 0.8, 13) # probability of performing each measurement in the brick wall
    p_us = [0] # probability of performing each unitary in the brick wall
    p_ms = [1] #np.linspace(0.4, 0.8, 9) # probability of performing each measurement in the brick wall

    for L in Ls:
        for p_u in p_us:
            for p_m in p_ms:
                timestamp = str(L)+','+str(p_m)+','+str(p_u)
                data_dir = 'data'#f'data_{timestamp}'
                os.makedirs(data_dir, exist_ok=True)
                print(f'calculating {L}, {p_u}, {p_m}')
                Parallel(n_jobs=-1)(delayed(evolve_state)(p_u, p_m, L, T(L), n, data_dir) for n in tqdm(range(N)))

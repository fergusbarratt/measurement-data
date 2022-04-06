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
    summer = qtn.tensor_gen.MPS_product_state([np.array([1.0, 1.0])] * L)
    psi0 = summer.copy(deep=True) / (summer.H @ summer)  # start from all + state

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

    astate = state(L, L//2-0)
    psi1 = qtn.MatrixProductState.from_dense(astate, [2]*L)

    bstate = state(L, L//2-1)
    psi2 = qtn.MatrixProductState.from_dense(bstate, [2]*L)

    return [final_state@psi1.H, final_state@psi2.H]#np.prod(final_state.norm)

def decode(record):
    L, T = record.shape
    T = (T-1)//2
    return np.argmax(evolve_state(0, L, T, 0, record.T))

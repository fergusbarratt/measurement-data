import quimb.tensor as qtn
import quimb as qu
import pickle
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path
import networkx as nx
from functools import reduce, lru_cache

from .tebdm import TEBDm
import gc

import joblib
plt.style.use('seaborn-whitegrid')

@lru_cache
def state(N, Q):
    psi = np.array([int(bin(n).count("1")==Q) for n in range(2**N)])
    return qu.qu(psi/np.sum(psi))

@lru_cache
def mps_state(N, Q):
    astate =  state(N, Q)
    return qtn.MatrixProductState.from_dense(astate, [2]*N)

def evolve_state(p, L, T, Q, record, unitaries=None, n=2):
    summer = qtn.tensor_gen.MPS_product_state([np.array([1.0, 1.0])] * L)
    psi0 = summer.copy(deep=True) / (summer.H @ summer)  # start from all + state

    tebd = TEBDm(p, psi0, H=np.ones((4, 4)))

    tebd.split_opts["cutoff"] = 0#1e-16 
    be_b = []

    try:
        for psit in tebd.at_times(np.arange(T), measurement_locations=record, biases=unitaries, progbar=False, tol=1e-3):    
            pass
    except ValueError:
        raise Exception
        # ValueError means the proposed set of biases and unitaries produce 
        # a vanishingly low probability for both options
        print('ValueError')
        return np.array([np.nan, np.nan])
    except ZeroDivisionError:
        print('ZeroDivisionError')
        # ZeroDivisionError the SVD controller got confused
        return np.array([np.inf, np.inf])
    
    final_state = tebd.pt
    final_state.norm = tebd.norm

    if n is None:
        n = L
    assert n is not None

    if n == 2:
        astate = state(L, L//2-0)
        psi1 = qtn.MatrixProductState.from_dense(astate, [2]*L)

        bstate = state(L, L//2-1)
        psi2 = qtn.MatrixProductState.from_dense(bstate, [2]*L)

        return np.prod(tebd.norm)*np.real(summer.H@summer*np.array([final_state@psi1.H, final_state@psi2.H]))
    else:
        res = []
        for q in range(L//2-n//2, L//2+n//2):
            #astate = state(L, q)
            psi = mps_state(L, q)#qtn.MatrixProductState.from_dense(astate, [2]*L)
            res.append(final_state@psi.H)
        return np.prod(tebd.norm)*np.real(summer.H@summer*np.prod(tebd.norm)*np.array(res))

def decode(record, unitaries=None, n=2):
    T, L = record.shape
    T = (T-1)//2+2
    logits = evolve_state(0, L, T, 0, record, unitaries if unitaries is not None else unitaries, n)
    return logits

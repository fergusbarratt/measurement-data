"""
functions for running non-symmetric random quantum circuits. 
"""
import numpy as np
import scipy.linalg as spla
import numba
from numba import njit, jit
import time
from functools import reduce, lru_cache

P0 = np.array([[1, 0], [0, 0]])
P1 = np.array([[0, 0], [0, 1]])

@lru_cache()
def state(N, Q):
    psi = np.array([int(bin(n).count("1")==Q) for n in range(2**N)])
    return psi/np.linalg.norm(psi)

@numba.jit(nopython=True)
def random_unitary():
    return np.linalg.qr(np.random.randn(4, 4)+np.random.randn(4, 4))[0]

def random_u1_unitary():
    A = np.linalg.qr(np.random.randn(2, 2)+1j*np.random.randn(2, 2))[0]
    return np.array([[1, 0, 0, 0], 
                     [0, A[0,0], A[0, 1], 0], 
                     [0, A[1, 0], A[1, 1], 0], 
                     [0, 0, 0,  np.exp(-1j*np.random.rand()*2*np.pi)]], dtype=np.complex128)

def apply_op(psi, site, op):
    N = int(np.log2(psi.shape[0]))
    n_sites = int(np.log2(op.shape[0]))

    reshaped_psi = psi.reshape(2**site, 2**n_sites, 2**(N-(site+n_sites))).transpose((1, 0, 2))
    shape = reshaped_psi.shape
    reshaped_psi = reshaped_psi.reshape(2**n_sites, -1)

    applied = np.dot(op, reshaped_psi)

    reshaped_psi = applied.reshape(shape).transpose((1, 0, 2)).reshape(-1)
    return reshaped_psi

def apply_random_unitaries(psi, even, random_unitary=random_u1_unitary, get_unitaries=False):
    N = int(np.log2(psi.shape[0]))
    batch = 2
    applied_weight = 0
    unitaries = []
    for site in range(even, N-1, batch*2):
        # get the batched op
        if site + batch*2 > N:
            leftover_batch = (N-site)//2
            new_unitaries = [random_unitary() for _ in range(leftover_batch)]
            unitaries = unitaries+new_unitaries
            op = reduce(np.kron, new_unitaries) if leftover_batch > 1 else random_unitary()
            applied_weight+=int(np.log2(op.shape[0]))
        else:
            new_unitaries = [random_unitary() for _ in range(batch)]
            unitaries = unitaries+new_unitaries
            op = reduce(np.kron, new_unitaries) if batch > 1 else random_unitary()
            applied_weight+=int(np.log2(op.shape[0]))

        psi = apply_op(psi, site, op)
    if get_unitaries:
        return psi, unitaries
    else:
        return psi

def apply_random_measurements(psi, sites_mask):
    N = int(np.log2(psi.shape[0]))
    sites = np.where(sites_mask!=0)[0]
    for site in sites:
        psi_0 = apply_op(psi, site, P0)
        measure_down = np.random.rand() < (psi.conj().T@psi_0).real #ev(psi, site, P0)

        sites_mask[site] = 2*measure_down-1
        if measure_down:
            psi = psi_0 # P0 has already been applied
        else:
            psi = psi-psi_0 # P1 = 1-P0

        psi /= np.linalg.norm(psi)
    return psi

def ev(psi, site, op):
    N = int(np.log2(psi.shape[0]))
    psi_ = apply_op(psi, site, op)
    return psi.conj().T@psi_

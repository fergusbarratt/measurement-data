"""
functions for running symmetric random quantum circuits. 
"""
import numpy as np
import scipy.linalg as spla
from scipy.special import binom
import scipy.linalg as spla
import numba
from abeliantensors import TensorU1

def state(N, Q):
    psi = TensorU1.initialize_with(np.ones, shape=[[1, 1]]*N, qhape=[[-1, 1]]*N, dirs=[1]*N, charge=Q)
    return psi

def random_u1_unitary(Uin=None):
    """
    Generate a random U1 unitary. 
    Args:
        Uin: The dense unitary to turn into a symmetric unitary. 
    """
    U = spla.block_diag(1, np.linalg.qr(np.random.randn(2, 2)+1j*np.random.randn(2, 2))[0], np.exp(-1j*np.random.rand()*2*np.pi))
    U = U if Uin is None else Uin
    tens = TensorU1.from_ndarray(U, shape=[[1, 2, 1], [1, 2, 1]], qhape=[[-2, 0, 2], [-2, 0, 2]], dirs=[1, -1], charge=0)
    return tens

def apply_two_site_op(psi, site, op):
    """
    """
    x, shape, qhape, dirs = psi.join_indices(site, site+1, return_transposed_shape_data=True)
    trans = list(range(1, site+1))+[0] + list(range(site+1, len(shape)-1))
    y = op.dot(x, [1, site]).transpose(trans)
    z = y.split_indices(site, shape[site:site+2], qhape[site:site+2], dirs[site:site+2])
    return z

    
    #N = np.log2(psi.shape[0]).astype(int)
    #if not site in list(range(N-1)):
    #    raise ValueError
    #operator = op
    #operand = (psi.reshape(2**(site), 4, 2**(N-(site+2))).transpose([1, 2, 0]))
    #return np.tensordot(operator, operand, [-1, 0]).transpose([2, 0, 1]).reshape(-1)

def apply_single_site_op(psi, site, op):
    N = np.log2(psi.shape[0]).astype(int)
    if not site in list(range(N)):
        raise ValueError
    return np.tensordot(op, psi.reshape(2**site, 2, 2**(N-site-1)).transpose([1, 2, 0]), [-1, 0]).transpose([2, 0, 1]).reshape(-1)

def ev(psi, site, op):
    N = np.log2(psi.shape[0]).astype(int)
    if not site in list(range(N)):
        raise ValueError

    psi_ = apply_single_site_op(psi, site, op)
    return psi.conj().T@psi_

def apply_random_unitaries(psi, even, random_unitary=random_u1_unitary):
    N = np.log2(psi.shape[0]).astype(int)
    for site in range(even, N-1, 2):
        op = random_unitary()
        psi = apply_two_site_op(psi, site, op)
    return psi

def apply_random_measurements(psi, sites_mask):
    N = np.log2(psi.shape[0]).astype(int)
    P0 = np.array([[1, 0], [0, 0]])
    P1 = np.array([[0, 0], [0, 1]])
    sites = np.where(sites_mask!=0)
    sites = sites[0] if len(sites) > 0 else []
    for site in sites:
        measure_down = np.random.rand() < ev(psi, site, P0)
        sites_mask[site] = 2*measure_down-1
        if measure_down:
            psi = apply_single_site_op(psi, site, P0)
            psi /= np.linalg.norm(psi)
        else:
            psi = apply_single_site_op(psi, site, P1)
            psi /= np.linalg.norm(psi)
    return psi

if __name__=='__main__':
    print('hello')
    print(cons_state(10, 5))

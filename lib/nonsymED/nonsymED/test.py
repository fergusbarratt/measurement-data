"""
Gets 
"""
import numpy as np
import scipy.linalg as spla
from joblib import Parallel, delayed
import time
import numba
import uuid
import argparse

def state(N, Q) :
    psi = np.array([int(bin(n).count("1")==Q) for n in range(2**N)])
    return psi/np.linalg.norm(psi)

@numba.jit(nopython=True)
def random_unitary():
    return np.linalg.qr(np.random.randn(4, 4)+np.random.randn(4, 4))[0]

def random_u1_unitary():
    return spla.block_diag(1, np.linalg.qr(np.random.randn(2, 2)+1j*np.random.randn(2, 2))[0], np.exp(-1j*np.random.rand()*2*np.pi))

def apply_two_site_op(psi, site, op):
    N = np.log2(psi.shape[0]).astype(int)
    if not site in list(range(N-1)):
        raise ValueError
    operator = op
    operand = (psi.reshape(2**(site), 4, 2**(N-(site+2))).transpose([1, 2, 0]))
    return np.tensordot(operator, operand, [-1, 0]).transpose([2, 0, 1]).reshape(-1)

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

def evolve_sector(N, Q, T, p):
    psi = state(N, Q)
    measurement_locs = np.random.choice([0, 1], size=(N, T), p=[1-p, p])
    measurement_locs = np.concatenate([measurement_locs, np.ones((N, 1))], axis=1)
    for t in range(T+1):
        print(f"{t}/{T}     \r", sep='', end='')
        psi = apply_random_unitaries(psi, 0, random_unitary=random_u1_unitary)
        psi = apply_random_unitaries(psi, 1, random_unitary=random_u1_unitary)
        psi = apply_random_measurements(psi, measurement_locs[:, t])
    return measurement_locs

def QQ_expm(M, N, p, multiprocessing=False):
    T = 2*N
    if multiprocessing:
        records_Q = Parallel(n_jobs=-1)(delayed(evolve_sector)(N, N//2, T, p) for _ in range(M))
        records_Q_plus_1 = Parallel(n_jobs=-1)(delayed(evolve_sector)(N, N//2+1, T, p) for _ in range(M))

    else:
        records_Q = [evolve_sector(N, N//2, T, p) for _ in range(M)]
        records_Q_plus_1 = [evolve_sector(N, N//2+1, T, p) for _ in range(M)]
    return np.stack([np.array(records_Q), np.array(records_Q_plus_1)], 0)

if __name__=='__main__':
    N_batches, batch_size = 180, 56
    parser = argparse.ArgumentParser(description="get some samples")
    parser.add_argument('L', type=int, help='system size')
    parser.add_argument('p', type=float, help='measurement probability')
    parser.add_argument('N_batches', type=int, help='number of batches', default=N_batches)
    args = parser.parse_args()
    L, p, N_batches = args.L, args.p, args.N_batches
    print(N_batches, L, p)
    for batch in range(1, N_batches+1):
        t = time.time()
        records = QQ_expm(56, L, p, True).transpose([1, 0, 2, 3])
        tim = time.time()-t
        print(f"got batch {batch}/{N_batches}, shape", records.shape, "in", tim, 's', f'approx remaining: {(N_batches-batch)*tim}s')
        np.save(f"test_data/{batch_size},{L},{p},{uuid.uuid4()}.npy", -records) # correct a problem with records

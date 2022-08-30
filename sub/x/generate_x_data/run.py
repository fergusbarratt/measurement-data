from nonsymED.nonsymed import *
import argparse
from joblib import Parallel, delayed
import joblib
import uuid
import time
import os

def state(N, Q):
    if Q == N//2:
        psi = [np.array([1, 0]), np.array([0, 1])]*(N//2)
        return reduce(np.kron, psi)
    elif Q == N//2+1:
        psi = [np.array([1, 0]), np.array([0, 1])]*(N//2)
        psi[N//2] = 1-psi[N//2]
        return reduce(np.kron, psi)
    else:
        raise Exception

def evolve_sector(N, Q, T, p):
    psi = reduce(np.kron, [np.array([1/np.sqrt(2), 1/np.sqrt(2)])]*N)
    measurement_locs = np.random.choice([0, 1], size=(N, T), p=[1-p, p])
    measurement_locs = np.concatenate([measurement_locs, np.ones((N, 1))], axis=1)
    for t in range(T+1):
        print(f"{t}/{T},", sep='', end='')
        psi = apply_random_unitaries(psi, t%2, random_unitary=random_u1_unitary)
        psi = apply_random_measurements(psi, measurement_locs[:, t])
    print()
    return measurement_locs

def QQ_expm(M, N, p, multiprocessing=True):
    T = 2*N
    if multiprocessing:
        records_Q = Parallel(n_jobs=-1)(delayed(evolve_sector)(N, N//2, T, p) for _ in range(M))
    else:
        records_Q = [evolve_sector(N, N//2, T, p) for _ in range(M)]
    return np.expand_dims(np.array(records_Q), 0)

def get_batches(L, p, N_batches, batch_size):
    os.makedirs('data', exist_ok=True)

    for batch in range(1, N_batches+1):
        t = time.time()
        records = QQ_expm(batch_size, L, p, True)
        print(records.shape)
        records = records.transpose([1, 0, 2, 3])
        tim = time.time()-t
        print(f"got batch {batch}/{N_batches}, shape", records.shape, "in", tim, 's', f'approx remaining: {(N_batches-batch)*tim}s')
        np.save(f"data/{batch_size},{L},{p},{uuid.uuid4()}.npy", -records) # correct a problem with records

if __name__=='__main__':
    total_samples = 10000
    batch_size = joblib.cpu_count()
    N_batches = total_samples//batch_size+1
    parser = argparse.ArgumentParser(description="get some samples")
    parser.add_argument('L', type=int, help='system size')
    parser.add_argument('p', type=float, help='measurement probability')
    parser.add_argument('N_batches', type=int, nargs='?', help='number of batches', default=N_batches)
    args = parser.parse_args()
    L, p, N_batches = args.L, args.p, args.N_batches
    print(N_batches, L, p, batch_size)
    get_batches(L, p, N_batches, batch_size)

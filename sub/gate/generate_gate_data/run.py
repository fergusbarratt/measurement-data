from code.nonsymed import *
import argparse
from joblib import Parallel, delayed
import joblib
import uuid
import time
import os

def evolve_sector(N, Q, T, p, get_unitaries=True):
    psi = state(N, Q)
    measurement_locs = np.random.choice([0, 1], size=(N, T), p=[1-p, p])
    measurement_locs = np.concatenate([measurement_locs, np.ones((N, 1))], axis=1)
    hops = np.zeros((N//2, T+1))
    for t in range(T+1):
        print(f"{t}/{T},", sep='', end='')
        psi, *unitaries = apply_random_unitaries(psi, t%2, random_unitary=random_u1_unitary, get_unitaries=get_unitaries)
        psi = apply_random_measurements(psi, measurement_locs[:, t])
        if get_unitaries:
            hops[t%2:, t] = np.array([np.abs(x[2, 1])**2 for x in unitaries[0]])
    print()
    if get_unitaries:
        return measurement_locs, hops
    else:
        return measurement_locs

def QQ_expm(M, N, p, multiprocessing=True, get_unitaries=True):
    T = 2*N
    if multiprocessing:
        records_Q = Parallel(n_jobs=56)(delayed(evolve_sector)(N, N//2, T, p, get_unitaries) for _ in range(M))
        records_Q_plus_1 = Parallel(n_jobs=56)(delayed(evolve_sector)(N, N//2+1, T, p, get_unitaries) for _ in range(M))
    else:
        records_Q = [evolve_sector(N, N//2, T, p, get_unitaries) for _ in range(M)]
        records_Q_plus_1 = [evolve_sector(N, N//2+1, T, p, get_unitaries) for _ in range(M)]
    if get_unitaries:
        return np.stack([np.array([x[0] for x in records_Q]), np.array([x[0] for x in records_Q_plus_1])], 0), np.stack([np.array([x[1] for x in records_Q]), np.array([x[1] for x in records_Q_plus_1])], 0), 
    else:
        return np.stack([np.array(records_Q), np.array(records_Q_plus_1)], 0)

def get_batches(L, p, N_batches, batch_size, get_unitaries=True):
    os.makedirs('data', exist_ok=True)

    for batch in range(1, N_batches+1):
        t = time.time()
        records, *unitaries = QQ_expm(batch_size, L, p, True)
        records = records.transpose([1, 0, 2, 3])
        tim = time.time()-t
        print(f"got batch {batch}/{N_batches}, shape", records.shape, "in", tim, 's', f'approx remaining: {(N_batches-batch)*tim}s')
        if get_unitaries:
            unitaries = unitaries[0].transpose([1, 0, 2, 3])
            np.savez(f"data/{batch_size},{L},{p},{uuid.uuid4()}.npz", unitaries=unitaries, records=-records)
        else:
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

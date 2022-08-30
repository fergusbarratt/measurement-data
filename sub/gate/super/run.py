from code.nonsymed import *
import argparse
from joblib import Parallel, delayed
import joblib
import uuid
from tqdm import tqdm
import time
import os
from measurement_analysis.stat_mech import decode
n_jobs = -1

def evolve_sector(N, Q, T, p, get_unitaries=True):
    psi = state(N, Q)
    measurement_locs = np.random.choice([0, 1], size=(N, T), p=[1-p, p])
    #measurement_locs = np.concatenate([measurement_locs, np.ones((N, 1))], axis=1)
    hops = np.zeros((N//2, T))
    for t in range(T):
        psi, *unitaries = apply_random_unitaries(psi, t%2, random_unitary=random_u1_unitary, get_unitaries=get_unitaries)
        psi = apply_random_measurements(psi, measurement_locs[:, t])
        if get_unitaries:
            hops[t%2:, t] = np.array([np.abs(x[2, 1])**2 for x in unitaries[0]])
    if get_unitaries:
        return measurement_locs, hops
    else:
        return measurement_locs

def QQ_expm(M, N, p, multiprocessing=True, get_unitaries=True):
    T = 2*N
    if multiprocessing:
        records_Q = Parallel(n_jobs=n_jobs)(delayed(evolve_sector)(N, N//2, T, p, get_unitaries) for _ in range(M))
        records_Q_plus_1 = Parallel(n_jobs=n_jobs)(delayed(evolve_sector)(N, N//2-1, T, p, get_unitaries) for _ in range(M))
    else:
        records_Q = [evolve_sector(N, N//2, T, p, get_unitaries) for _ in range(M)]
        records_Q_plus_1 = [evolve_sector(N, N//2-1, T, p, get_unitaries) for _ in range(M)]
    if get_unitaries:
        return np.stack([np.array([x[0] for x in records_Q]), np.array([x[0] for x in records_Q_plus_1])], 0), \
               np.stack([np.array([x[1] for x in records_Q]), np.array([x[1] for x in records_Q_plus_1])], 0), 
    else:
        return np.stack([np.array(records_Q), np.array(records_Q_plus_1)], 0)

def decode_logits(record, unitaries=None):
    unbiased_unitaries = 0*unitaries+np.ones(unitaries.shape)/2
    antibiased_unitaries = 1-unitaries

    biased_result = decode(record, unitaries).real
    unbiased_result = decode(record, unbiased_unitaries).real
    antibiased_result = decode(record, antibiased_unitaries).real

    result = np.stack([unbiased_result, biased_result, antibiased_result], axis=0) # (biased, logits)
    return result 

def get_batch(L, p, batch_size, get_unitaries=True):
    os.makedirs('data', exist_ok=True)

    print('creating')
    t = time.time()
    records, *unitaries = QQ_expm(batch_size, L, p, True)
    records = records.transpose([1, 0, 2, 3])
    unitaries = unitaries[0].transpose([1, 0, 2, 3])
    tim = time.time()-t
    print(records.shape, unitaries.shape)

    print('decoding')
    logits_0 = np.array(Parallel(n_jobs=n_jobs)(delayed(decode_logits)(record.T, unitary) for record, unitary in tqdm(list(zip(records[:, 0], unitaries[:, 0])))))
    logits_1 = np.array(Parallel(n_jobs=n_jobs)(delayed(decode_logits)(record.T, unitary) for record, unitary in tqdm(list(zip(records[:, 1], unitaries[:, 1])))))
    logits = np.stack([logits_0, logits_1])
    logits = logits.transpose([1, 2, 0, 3]) # (sample, bias, expected, predicted)
    return logits

if __name__=='__main__':
    Ls = [8, 10, 12, 14, 16, 18]
    ps = [0.025, 0.05, 0.075, 0.1, 0.125, 0.15,  0.175, 0.2,  0.225, 0.25, 0.275, 0.3, 0.325, 0.35, 0.375, 0.4]
    np.save('data/Ls', np.array(Ls))
    np.save('data/ps', np.array(ps))

    N = 5000
    H = np.zeros((4, 4))
    ϵ = 1e-10
    T = lambda L: L
    all_logits = []
    for L in Ls:
        p_logits = []
        for p in ps:
            print('calculating')
            if os.path.exists(f'data/checkpoints/{L}_{p}.npy'):
                logits = np.load(f'data/checkpoints/{L}_{p}.npy')
            else:
                logits = get_batch(L, p, N)
                np.save(f'data/checkpoints/{L}_{p}.npy', logits)
            p_logits.append(np.array(logits))
        all_logits.append(np.array(p_logits))
    np.save(f'data/raw_data_{N}_{uuid.uuid4()}.npy', np.array(all_logits))
    print('done')

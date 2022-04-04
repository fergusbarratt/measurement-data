from functools import reduce

from joblib import Parallel, delayed
import joblib
import socket
import psutil
import time
from itertools import product
import os
from measurement_analysis.stat_mech import decode
import numpy as np
from tqdm import tqdm
data_loc = '../processed_data/'

def get_host(a=None, b=None):
    core = psutil.Process().cpu_num()
    cpu_count = joblib.cpu_count()
    def f():
        time.sleep(5e-1)
        return psutil.Process().cpu_num()
    ret = sorted(list(Parallel(n_jobs=-1)(delayed(f)() for _ in range(56))))
    return socket.gethostname(), cpu_count, ret

def decode_records(records):
    q0s = Parallel(n_jobs=-1)(delayed(decode)(record) for record in tqdm(records[:, 0]))
    q1s = Parallel(n_jobs=-1)(delayed(decode)(record) for record in tqdm(records[:, 1]))
    a1 = 1-np.mean(q0s)
    a2 = np.mean(q1s)
    return (a1+a2)/2

if __name__ == "__main__":
    print('running')
    # Put the datasets on the host process
    Ls = [10, 12, 14, 16, 18]
    ps = [0.05, 0.075, 0.1, 0.125, 0.15, 0.25, 0.35, 0.45, 0.5]
    K = 20

    # read in the data
    all_data = {}
    for L in Ls:
        for p in ps:
            def end(L):
                return int(6*np.sqrt(L)+1)
            arr =  np.expand_dims(np.load(data_loc+f"{L},{p}.npy")[:10000, :, :, :end(L)], 0) # drop last (fully measured) row
            #np.save(f"processed_data/{L},{p}.npy", arr)(-1)**(L==16 and p < 0.5 and p > 0.05)*
            all_data[(str(L), str(p))] = arr
            
    # stack the data for different p
    collated = {}
    for key, value in sorted(all_data.items()):
        if key[0] in collated:
            collated[key[0]] = np.concatenate([collated[key[0]], value], axis=0)
        else:
            collated[key[0]] = value 

    print("collated")
    for key, x in collated.items():
        print(key, x.shape)

    K = 10000
    pLs = sorted(list(product(Ls, range(len(ps)))))
    print(len(pLs))

    #records = [collated[str(L)][p_ind, :K] for L, p_ind in pLs] # all the sets of L, p, records
    import argparse
    parser = argparse.ArgumentParser(description='..')
    parser.add_argument('L', type=int)
    parser.add_argument('p_ind', type=int)
    args = parser.parse_args()
    records = collated[str(args.L)][args.p_ind, :K]
    print(records.shape)

    path = f'../notebooks/measurement_analysis/notebooks/data/statmechdecode_10,{args.L},{args.p_ind}.npy'
    if not os.path.exists(path):
        results = decode_records(records)
        print(results)
        np.save(path, np.array(results))
    else:
        print('already computed')


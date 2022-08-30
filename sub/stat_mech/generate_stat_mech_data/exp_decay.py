import quimb.tensor as qtn
import quimb as qu
import os
import shutil
import numpy as np
import itertools
import matplotlib.pyplot as plt
import time
from functools import reduce
from tqdm import tqdm

from tebdm import TEBDm
import gc

from dask.distributed import Client, progress, LocalCluster
from dask_jobqueue import LSFCluster
from dask import delayed
import dask
import joblib
plt.style.use('seaborn-whitegrid')

import logging

def evolve_state(p, L, T, n, data_dir):
    summer = qtn.tensor_gen.MPS_product_state([np.array([1.0, 1.0])] * L)
    psi0 = summer.copy(deep=True) / (summer.H @ summer)  # start from all + state

    tebd = TEBDm(p, psi0, H, imag=True)

    tebd.split_opts["cutoff"] = 1e-8
    be_b = []
    for psit in tebd.at_times(np.arange(T), progbar=False, tol=1e-3):    
        be_b += [psit.entropy(L//2)]    
    
    final_state = tebd.pt
    final_state.entanglement_history = np.array(be_b)
    final_state.norm = tebd.norm
    
    qid = qtn.tensor_core.rand_uuid()
    qu.save_to_disk(final_state, data_dir + f'/{p},{L},{T},{qid}.dmp')
    del final_state
    del summer
    del psi0
    del tebd
    del be_b
    gc.collect()
    return None

if __name__ == '__main__':
    Ls = [60]#[16, 20, 24, 28, 32]
    N = 64
    #ps = [0.05, 0.1, 0.15]+[x for x in np.linspace(0.2, 0.4, 5)]+[0.45, 0.5]
    ps = [0.5]#[0.05, 0.1, 0.15, 0.2, 0.225, 0.25, 0.275, 0.3, 0.325, 0.35, 0.375, 0.4]
    np.random.shuffle(ps) # get the ps in a random order
    np.random.shuffle(Ls)
    H = np.zeros((4, 4))
    ϵ = 1e-10
    T = lambda L: 2*L
    timestamp = "exp_decay"#time.time()
    qid = qtn.tensor_core.rand_uuid()
    root = '/tmp/fergus_'+ qid +'/'
    data_dir = f'data_{timestamp}'
    total_dir = root+data_dir
    print(total_dir)
    os.makedirs(root+data_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)

    par = 'joblib'
    if par == 'dask':
        cluster = LSFCluster(queue='long',
                             walltime='8:00',
                             cores=32,
                             memory='128GiB', 
                             processes=32,
                             job_extra=['-R select[rh=8]', '-R rusage[tmp=10000]'],
                             local_directory=os.getenv('TMPDIR', '/tmp'),
                             extra=["--lifetime", "400m", "--lifetime-stagger", "20m"],
                             interface='ib0',
                             use_stdin=True)
        cluster.adapt(minimum_jobs=1, maximum_jobs=3)

        client = Client(cluster)
        cluster.scheduler.allowed_failures = 100
        all_delayed = []
        for L in Ls[::-1]:
            for p in ps:
                for n in range(N):
                    all_delayed.append(delayed(evolve_state)(p, L, T(L), n, total_dir))

        from dask.distributed import progress
        print('calculating...')

        x = dask.compute(all_delayed)[0][0]
        print('done.')

    elif par == 'joblib':
        for L in Ls:
            for p in ps:
                print('calculating')
                joblib.Parallel(n_jobs=-1)(joblib.delayed(evolve_state)(p, L, T(L), n, data_dir) for n in tqdm(range(N)))
        print('done, moving folder')
        shutil.move(total_dir, data_dir)
        print('moved')

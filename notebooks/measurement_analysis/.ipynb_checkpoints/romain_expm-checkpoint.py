#!/usr/bin/env python
# coding: utf-8

# # Machine learning U(1) sharpening

# # Imports, plus data loading

# ## Imports

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from functools import reduce
from tqdm.auto import tqdm
from itertools import product
from joblib import Parallel, delayed
data_loc = '../../../processed_data/'
fig_loc = '/home1/08522/barratt/figs/u1/'
out_data_loc = './data/'
plt.style.use('seaborn-whitegrid')
decode=True
percolate=False
unmeasured_percolate=False
mpi=True
if mpi:
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()
else:
    rank = 0


# ## Labelling data + preprocessing

# In[2]:

if rank==0:

    Ls = [10, 12, 14, 16, 18]
    ps = [0.05, 0.075, 0.1, 0.125, 0.15, 0.25, 0.35, 0.45, 0.5]

    # read in the data
    all_data = {}
    for L in Ls:
        for p in ps:
            arr =  np.expand_dims(np.load(data_loc+f"{L},{p}.npy")[:10000, :, :, :2*L], 0) # drop last (fully measured) row
            #np.save(f"processed_data/{L},{p}.npy", arr)(-1)**(L==16 and p < 0.5 and p > 0.05)*
            all_data[(str(L), str(p))] = arr
            
    # stack the data for different p
    collated = {}
    for key, value in sorted(all_data.items()):
        if key[0] in collated:
            collated[key[0]] = np.concatenate([collated[key[0]], value], axis=0)
        else:
            collated[key[0]] = value 

    # get Q/Q+1 labels, shuffle both labels and images, store both, drop last (fully measured) row
    datasets = {}
    for key, value in collated.items():
        labels = np.repeat(np.expand_dims(np.concatenate([np.zeros((collated[key].shape[1], 1)), np.ones((collated[key].shape[1], 1))], axis=0), 0), len(ps), axis=0)
        datasets[key] = np.expand_dims(np.concatenate([collated[key][:, :, i, :, :] for i in [0, 1]], axis=1), -1).astype(float) # add dummy channels, float

        perm = np.arange(labels.shape[1])
        np.random.shuffle(perm)
        labels = labels[:, perm]
        datasets[key] = (datasets[key][:, perm], labels)
        
    print("datasets")
    for key, (x_train, y_train) in datasets.items():
        print(key, x_train.shape, y_train.shape)
    print("collated")
    for key, x in collated.items():
        print(key, x.shape)

    # # Percolation

    # ## Algorithm

    # In[3]:

    from measurement_analysis.percolation import is_percolation, find_clusters_bond


    # ## Analysis on sharp sites

    # A typical trajectory. We have a bond percolation problem on a tilted square lattice. 

    # In[4]:


    record = collated['18'][-2, 5, 0].T 
    abs_record = np.abs(record)
    clusters, ids = find_clusters_bond(abs_record)
    percolation = is_percolation(ids)

    fig, ax = plt.subplots(1, 3, dpi=130, figsize=(8, 8))
    ax[0].set_title(f'site clusters: \n percolation: {percolation}')
    ax[1].set_title('blocked bonds')
    ax[2].set_title('trajectory')

    ax[0].pcolormesh(ids, edgecolors='k', linewidth=2, cmap='jet')
    ax[1].pcolormesh(1-abs_record, edgecolors='w', linewidth=2, cmap='bone_r')
    ax[2].pcolormesh(record, edgecolors='w',cmap='bwr', vmin=-1, vmax=1)

    ax[0].invert_yaxis()
    ax[1].invert_yaxis()
    ax[2].invert_yaxis();
    plt.savefig(fig_loc + 'percolation_clusters.png')


    # ### Locating the percolation transition

    # In[10]:

    if percolate:
        print('Locating the percolation transition')

        all_perc_probs = []
        for L in tqdm([str(L) for L in Ls]):
            perc_probs = []
            K = None
            for p_ind in tqdm(range(0, 9)):
                def get_perc(record):
                    record = record.T
                    record = record[:min(record.shape), :min(record.shape)]
                    abs_record = np.abs(record)
                    clusters, ids = find_clusters_bond(abs_record)
                    return is_percolation(ids)
                percolations = Parallel(n_jobs=-1)(delayed(get_perc)(record) for record in tqdm(collated[L][p_ind, :K, 0], position=0, leave=True))
                perc_probs.append(np.mean(percolations))
            all_perc_probs.append(perc_probs)

        np.save(out_data_loc + 'percolation_transition.npy', np.array(all_perc_probs))

        # In[11]:


        plt.figure(dpi=120)
        for i, perc_probs in enumerate(all_perc_probs):
            plt.plot(ps, perc_probs, label=str(Ls[i]))
        plt.legend()
        plt.ylim([-1e-2, 1]);
        plt.ylabel('percolation probability')
        plt.xlabel('measurement probability');
        plt.savefig(fig_loc + 'percolation_probability.png')

    # ## Analysis on sharp sites + unmeasured sharp sites
    # Fill out unsharp sites using information from sharp sites

    # ![image.png](attachment:9e60e71d-c3dc-4200-b747-8bb3be0a0204.png)

    # ### Algorithm

    # In[12]:


    integer = np.random.randint(0, 500)


    # In[13]:


    from measurement_analysis.unmeasured_percolation import iterated_fill_out, fill_out
    from measurement_analysis.graph import get_graph


    # ### Analysis

    # In[14]:


    record = collated['18'][-1, np.random.randint(0, 50), 0][:min(record.shape), :min(record.shape)].T 

    abs_record = np.abs((np.copy(record)))
    filled_abs_record = np.abs(fill_out(np.copy(record)))
    filled_abs_record2 = np.abs(fill_out(fill_out(np.copy(record))))

    clusters, ids = find_clusters_bond(filled_abs_record2)
    percolation = is_percolation(ids)

    fig, ax = plt.subplots(1, 5, dpi=130, figsize=(8, 3), sharey=True)
    ax[0].set_title(f'site clusters: \n percolation: {percolation}')
    ax[1].set_title('blocked bonds \n after filling out twice')

    ax[2].set_title('blocked bonds \n after filling out')
    ax[3].set_title('blocked bonds')
    ax[4].set_title('trajectory')

    ax[0].pcolormesh(ids, edgecolors='k', linewidth=2, cmap='jet')
    ax[2].pcolormesh(1-filled_abs_record, edgecolors='w', linewidth=2, cmap='bone_r')
    ax[1].pcolormesh(1-filled_abs_record2, edgecolors='w', linewidth=2, cmap='bone_r')
    ax[3].pcolormesh(1-abs_record, edgecolors='w', linewidth=2, cmap='bone_r')
    ax[4].pcolormesh(record, edgecolors='w',cmap='bwr', vmin=-1, vmax=1)

    ax[0].invert_yaxis()
    ax[1].invert_yaxis()
    ax[2].invert_yaxis()
    ax[3].invert_yaxis()
    ax[3].invert_yaxis();

    fig.tight_layout()

    plt.savefig(fig_loc + 'filling_out.png')


    # ### Locating the unmeasured sharp site percolation transition

    # In[167]:

    if unmeasured_percolate:
        print('Locating the unmeasured percolation transition')

        all_perc_probs = []
        for L in tqdm([str(L) for L in Ls]):
            perc_probs = []
            K = None
            for p_ind in tqdm(range(0, 9), leave=False):
                def get_perc(record):
                    record = record.T
                    record = record[:min(record.shape), :min(record.shape)]
                    abs_record = np.abs(iterated_fill_out(np.copy(record))) # fill out the record using local rules, and get the absolute value

                    clusters, ids = find_clusters_bond(abs_record)
                    return is_percolation(ids)
                percolations = Parallel(n_jobs=-1)(delayed(get_perc)(record) for record in tqdm(collated[L][p_ind, :K, 0], position=0, leave=True))
                perc_probs.append(np.mean(percolations)) # What fraction of the time do we get a percolating cluster for this p
            all_perc_probs.append(perc_probs)

        np.save(out_data_loc + 'unmeasured_percolation_transition.npy', np.array(all_perc_probs))

        # In[ ]:


        plt.figure(dpi=120)
        for i, perc_probs in enumerate(all_perc_probs):
            plt.plot(ps, (1/2*(1+np.array(perc_probs))), label=str(Ls[i]))
        plt.legend()
        plt.ylim([-1e-2, 1]);
        plt.ylabel('percolation probability')
        plt.xlabel('measurement probability');

        plt.savefig(fig_loc + 'unmeasured_percolation_probability.png')

    # ## Stat. Mech

    # In[145]:


    from measurement_analysis.stat_mech import decode
    from pathlib import Path


    # In[122]:


    record = collated['18'][-1, np.random.randint(0, 50), 0]
    decode(record)


# In[161]:

if decode:
    if not mpi:
        print('Locating the stat. mech. decoding transition')
        all_accs = []
        for L in tqdm([str(L) for L in Ls]):
            accs = []
            K = 1000
            for p_ind in tqdm(range(0, 9), leave=False):
                q0s = Parallel(n_jobs=-1)(delayed(decode)(record) for record in collated[L][p_ind, :K, 0])
                q1s = Parallel(n_jobs=-1)(delayed(decode)(record) for record in collated[L][p_ind, :K, 1])
                a1 = 1-np.mean(q0s)
                a2 = np.mean(q1s)
                accs.append((a1+a2)/2)
            all_accs.append(accs)

        np.save(out_data_loc + 'stat_mech_decoding.npy', np.array(all_accs))

        # In[163]:


        plt.figure(dpi=120)
        plt.plot(ps, np.array(all_accs).T, label=Ls)
        plt.legend();

        plt.savefig(fig_loc + 'decoding_accuracy.png')
    else:
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        size = comm.Get_size()
        rank = comm.Get_rank()
        print(mpi, size, rank)

        if rank == 0:
            print('Locating the stat. mech. decoding transition')

        def get_acc(L, p_ind):
            q0s = Parallel(n_jobs=-1)(delayed(decode)(record) for record in collated[L][p_ind, :K, 0])
            q1s = Parallel(n_jobs=-1)(delayed(decode)(record) for record in collated[L][p_ind, :K, 1])
            a1 = 1-np.mean(q0s)
            a2 = np.mean(q1s)
            return (a1+a2)/2
        raise Exception

import numpy as np
import matplotlib.pyplot as plt
import os
import pathlib
plt.style.use('seaborn-whitegrid')
figs_dir = os.environ['HOME']+'/figs'
os.makedirs(figs_dir, exist_ok=True)

L = 16
arr1 = np.load(f'processed_data/0.1,{L},{4*L},get_temporal_entanglement.npy')
arr2 = np.load(f'processed_data/0.9,{L},{4*L},get_temporal_entanglement.npy')
e1 = np.mean(arr1, axis=0)
e2 = np.mean(arr2, axis=0)
plt.plot(list(range(1, len(e1)+1)), e1, marker='x', label='mostly unitaries ($p_m=0.1$)')
plt.plot(list(range(1, len(e2)+1)), e2, marker='x', label='mostly measurements ($p_m = 0.9$)')
plt.legend()
plt.xscale('log')
plt.show()


#arr = np.load('processed_data/0.0,1.0,8,32,get_correlations.npy')
arr1 = np.load(f'processed_data/0.1,{L},{4*L},get_correlations.npy')
arr2 = np.load(f'processed_data/0.9,{L},{4*L},get_correlations.npy')
#arr = np.load('processed_data/1.0,1.0,8,32,get_correlations.npy')
a, b = L//4, 3*L//4 
a, b = None, None
C = arr1[:, :-1, :]
z = arr1[:, -1:, :]
c11 = np.mean(C, axis=0)[a:b, a:b]
c21 = np.mean(z*np.transpose(z, [0, 2, 1]), axis=0)[a:b, a:b]
corrs1 = c11-c21

C = arr2[:, :-1, :]
z = arr2[:, -1:, :]
c12 = np.mean(C, axis=0)[a:b, a:b]
c22 = np.mean(z*np.transpose(z, [0, 2, 1]), axis=0)[a:b, a:b]
corrs2 = c12+c22
np.set_printoptions(precision=3)
c1s = [c11, c12]
c2s = [c21, c22]
corrs = [corrs1, corrs2]

fig, ax = plt.subplots(1, 2)
ax[0].imshow(np.abs(c2s[0]))
ax[1].imshow(np.abs(c2s[1]))
plt.show()

def g(corrs):
    gs = []
    rs = list(range(1, corrs.shape[0]))
    for r in rs:
        v = np.diag(corrs, k=r)
        #gs.append(np.mean(v))
        gs.append(v[0])
    return (rs, gs)


fig, ax = plt.subplots(2, 2, sharex='col', sharey='row')
for i in [0, 1]:
    ax[0][i].plot(*g(c1s[i]), marker='x')
    ax[0][i].axhline(-1/(L-1), linestyle='--', c='black', label='$1/(L-1)$')
    ax[1][i].plot(*g(c2s[i]), marker='x')
    ax[1][i].set_title("$-\\langle Z\\rangle \\langle Z\\rangle $", loc='right')
ax[0][0].set_title("p_m = 0.1            $\\langle ZZ\\rangle $", loc='right')
ax[0][1].set_title("p_m = 0.9            $\\langle ZZ\\rangle $", loc='right')
plt.tight_layout()
plt.show()
raise Exception

######################################
#         MEASUREMENT ONLY           #
######################################

arrs = {str(x.stem):np.load(str(x)) for x in pathlib.Path('cache_data/clean/processed_data').glob('**/*get_temporal_entanglement*.npy') if not str(x.stem).startswith('file')}
arrs = [x[1] for x in sorted([(len(x[1]), x) for x in list(arrs.values())])]
mean_arrs = [arr.mean(axis=0) for arr in arrs]

fig, ax = plt.subplots(1, 4, figsize=(10, 4))
for x in mean_arrs:
    ax[0].plot(x, label=str(len(x)//2))
    ax[1].plot(x, label=str(len(x)//2))
    ax[1].set_xscale('log')
    ax[2].plot(x[1:]-x[1], label=str(len(x)//2))
    ax[3].plot(x[1:]-x[1], label=str(len(x)//2))
    ax[3].set_xscale('log')

ax[0].legend()
ax[2].legend()
ax[3].legend()
fig.tight_layout()
plt.savefig(figs_dir+'/entanglement_vs_t.pdf')

fig, ax = plt.subplots(1, 2, figsize=(7, 4), dpi=150)

k = 3
max_cut = np.min([len(x) for x in mean_arrs[k:]])-2
cut = -1

x_Ls = [(x[1:]-x[1])[int(len(x)//4)] for x in mean_arrs[k:]]
ax[0].plot([len(x)//2 for x in mean_arrs[k:]], x_Ls, marker='.')

x_Ls = [(x[1:]-x[1])[max_cut] for x in mean_arrs[k:]]
ax[1].plot([len(x)//2 for x in mean_arrs[k:]], x_Ls, marker='.')

ax[0].set_xlabel('$L$')
ax[0].set_ylabel('$S_E(L)$')

ax[1].set_ylabel(f'$S_E({max_cut})$')
ax[1].set_xscale('log')
ax[1].set_xlabel('$L$')


fig.tight_layout()
plt.savefig(figs_dir+'/entanglement_vs_L.pdf')

rescaled_mean_arrs = [x[1:]-x[1] for x in mean_arrs]

from scipy.stats import linregress

fig, ax = plt.subplots(1, 2, figsize=(7, 4), dpi=150)
Ts = (np.arange(4, 32, 2)*2)
q = 1/2
fits = []
for i, (y, T) in enumerate(zip(rescaled_mean_arrs, Ts)):
    ax[0].scatter(np.arange(1, int(T*q)), y[:int(T*q)-1], marker='.', c=f'C{i}')
    fit = linregress(np.log(np.arange(1, int(T*q))), y[:int(T*q)-1])
    fits.append(fit)
    ax[0].plot(np.arange(1, int(T*q)), fit.slope*np.log(np.arange(1, int(T*q)))+fit.intercept, c=f'C{i}', linestyle='--')

Ls = Ts/2
ax[1].plot(Ls, np.array([fit.slope for fit in fits]), marker='.')
ax[1].set_ylabel('slope')
ax[0].set_ylabel('$S_E(t)$')

ax[0].set_xlabel('$t$')
ax[1].set_xlabel('$L$')

ax[0].set_xscale('log')
fig.tight_layout()
plt.savefig(figs_dir+'/analyse.pdf')

########################################################
#          WITH (randomly placed) UNITARIES            #
########################################################

arrs = {str(x.stem):np.load(str(x)) for x in pathlib.Path('cache_data/unitaries/processed_data').glob('**/*get_temporal_entanglement*.npy') if not str(x.stem).startswith('file')}
#print(sorted([(float(x.split(',')[0]), x) for x in list(arrs.keys())]))
arrs = [x[1] for x in sorted([(float(x.split(',')[0]), arrs[x]) for x in list(arrs.keys())])]
mean_arrs = [arr.mean(axis=0) for arr in arrs]

fig, ax = plt.subplots(1, 2, figsize=(7, 4), dpi=150, sharey=True)
ps = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
for p, x in zip(ps, mean_arrs):
    ax[0].plot(x, marker='.', label='{:.2f}'.format(p))
    ax[1].plot(x[1:], marker='.', label='{:.2f}'.format(p))
    ax[1].set_xscale('log')
ax[0].legend()
ax[1].legend()
ax[0].set_xlabel('$t$')
ax[1].set_xlabel('$t$')
ax[0].set_ylabel('$S_E(t)$')
fig.tight_layout()
plt.savefig(figs_dir+'/w_unitaries.pdf')

#########################################################################
#          WITH UNITARIES and (randomly placed MEASUREMENTS)            #
#########################################################################

arrs = {str(x.stem):np.load(str(x)) for x in pathlib.Path('processed_data').glob('**/*get_temporal_entanglement*.npy') if not str(x.stem).startswith('file')}
ps = sorted([float(str(x).split(',')[1]) for x in arrs.keys()])
arrs = [x[1] for x in sorted([(float(x.split(',')[1]), arrs[x]) for x in list(arrs.keys())])]
mean_arrs = [arr.mean(axis=0) for arr in arrs]

fig, ax = plt.subplots(1, 2, figsize=(7, 4), dpi=150, sharey=True)

colors = plt.cm.Spectral_r(np.linspace(0,1,len(mean_arrs)))

for i, (p, x) in enumerate(zip(ps, mean_arrs)):
    ax[0].plot(x, marker='.', label='{:.2f}'.format(p), c=colors[i])
    ax[1].plot(x[1:], marker='.', label='{:.2f}'.format(p), c=colors[i])
    ax[1].set_xscale('log')
ax[0].legend()
ax[1].legend()
ax[0].set_xlabel('$t$')
ax[1].set_xlabel('$t$')
ax[0].set_ylabel('$S_E(t)$')
fig.tight_layout()
plt.savefig(figs_dir+'/w_unitaries_random_measurements.pdf')

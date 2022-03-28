import numpy as np
import matplotlib.pyplot as plt
import os
import pathlib
plt.style.use('seaborn-whitegrid')
figs_dir = os.environ['HOME']+'/figs'
os.makedirs(figs_dir, exist_ok=True)

arr = np.load('processed_data/0.0,1.0,8,32,get_temporal_entanglement.npy')
e = np.mean(arr, axis=0)
plt.plot(list(range(1, len(e)+1)), e, marker='x')
plt.xscale('log')
plt.show()


arr = np.load('processed_data/0.0,1.0,8,32,get_correlations.npy')
C = arr[:, :-1, :]
z = arr[:, -1:, :]
c1 = np.mean(C, axis=0)
c2 = np.mean(z*np.transpose(z, [0, 2, 1]), axis=0)
corrs = c1-c2

np.set_printoptions(precision=3)
#corrs = np.mean(C, axis=0)[4:12, 4:12]
plt.imshow(np.abs(corrs), vmin=0, vmax=0.2)
plt.colorbar()
plt.show()

def g(corrs):
    gs = []
    rs = list(range(1, corrs.shape[0]))
    for r in rs:
        v = np.diag(corrs, k=r)
        print(v)
        gs.append(np.mean(v))
    return (rs, gs)


fig, ax = plt.subplots(3, 1, sharex=True)
ax[0].plot(*g(c1), marker='x')
ax[0].set_title("<ZZ>", loc='right')
ax[1].plot(*g(c2), marker='x')
ax[1].set_title("<Z><Z>", loc='right')
ax[2].plot(*g(corrs), marker='x')
ax[2].set_title("<ZZ>-<Z><Z>", loc='right')
plt.tight_layout()
plt.show()
print(corrs.shape)
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

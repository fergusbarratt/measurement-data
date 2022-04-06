import numpy as np
import matplotlib.pyplot as plt
from scipy import special

comb = special.comb
kl = special.rel_entr


def f(s, N, Q, p):
    return comb(int(p*N), s)*comb(int((1-p)*N), Q-s) / comb(N, Q)


p = 0.5
for N in [50, 100, 150, 200, 1000]:
    ss = np.arange(1, N//2)
    plt.plot(ss/N, [f(s, N, N//2, p) for s in ss], label=f'{N}')
    plt.plot(ss/N, [f(s, N, N//2+1, p) for s in ss], label=f'{N}')
    #plt.plot(ss, [f(s, N, N//2+1, p) for s in ss], label='N/2')

plt.legend()
plt.show()

all_fs_Q = []
all_fs_Q_1 = []
Ns = np.arange(50, 1000, 50)
for N in Ns:
    ss = np.arange(1, N//2)
    all_fs_Q.append(np.array([f(s, N, N//2, p) for s in ss]))
    all_fs_Q_1.append(np.array([f(s, N, N//2+1, p) for s in ss]))

divs = []
for fs_Q, fs_Q_1 in zip(all_fs_Q, all_fs_Q_1):
    divs.append(np.sum(kl(fs_Q, fs_Q_1)))
plt.plot(Ns, divs)
plt.show()

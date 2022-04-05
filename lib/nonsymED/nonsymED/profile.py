from nonsymED.nonsymed import *
import time

def evolve_sector(N, Q, T, p):
    psi = state(N, Q)
    measurement_locs = np.random.choice([0, 1], size=(N, T), p=[1-p, p])
    measurement_locs = np.concatenate([measurement_locs, np.ones((N, 1))], axis=1)
    for t in range(T+1):
        #print(f"{t}/{T}     \r", sep='', end='')
        psi = apply_random_unitaries(psi, t%2, random_unitary=random_u1_unitary)
        psi = apply_random_measurements(psi, measurement_locs[:, t])
    return measurement_locs

if __name__=='__main__':
    N = 2
    t0 = time.time()
    for _ in range(N):
        evolve_sector(20, 10, 10, 0.0)
    t1 = time.time()
    print((t1-t0)/N)

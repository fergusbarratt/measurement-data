import joblib
from run import get_batches

if __name__=='__main__':
    total_samples = 1000
    batch_size = joblib.cpu_count()
    N_batches = total_samples//batch_size+1

    #Ls = [20]
    #ps = [0.025, 0.0375, 0.05, 0.0625, 0.075, 0.0875, 0.1, 0.125, 0.15, 0.2, 0.25, 0.35, 0.45, 0.5]
    #pLs = zip(Ls, ps)

    #pLs = [(20, 0.5), (20, 0.0625), (20, 0.0375), (20, 0.045), (20, 0.035), (20, 0.025), (20, 0.25)]
    pLs = [(18, 0.0875), (16, 0.0875), (16, 0.2), (18, 0.025), (18, 0.0375)]

    for (L, p) in pLs:
        get_batches(L, p, N_batches, batch_size)

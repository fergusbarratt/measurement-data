import joblib
from run import get_batches

if __name__=='__main__':
    total_samples = 10000
    batch_size = joblib.cpu_count()
    N_batches = total_samples//batch_size+1

    Ls = [14, 16, 18, 20]
    ps = [0.025]
    for L in Ls:
        for p in ps:
            get_batches(L, p, N_batches, batch_size)

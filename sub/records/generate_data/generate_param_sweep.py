"""
"""
import os
import argparse
if os.system('which bsub') == 0:
    cluster = 'umass'
else:
    cluster = 'frontera'
import pickle

targets = pickle.load(open('remaining', 'rb'))#{(6, 0.025): 16, (6, 0.0375): 16, (6, 0.05): 16, (6, 0.0625): 16, (6, 0.075): 16, (6, 0.0875): 16, (6, 0.1): 16, (6, 0.125): 16, (6, 0.15): 16, (6, 0.2): 1640, (6, 0.25): 16, (6, 0.3): 20000, (6, 0.35): 16, (6, 0.45): 16, (6, 0.5): 16, (8, 0.025): 16, (8, 0.0375): 16, (8, 0.05): 16, (8, 0.0625): 16, (8, 0.075): 16, (8, 0.0875): 16, (8, 0.1): 16, (8, 0.125): 16, (8, 0.15): 16, (8, 0.2): 1640, (8, 0.25): 16, (8, 0.3): 20000, (8, 0.35): 16, (8, 0.45): 16, (8, 0.5): 0, (10, 0.025): 16, (10, 0.0375): 8, (10, 0.05): 20, (10, 0.0625): 40, (10, 0.075): 40, (10, 0.0875): 40, (10, 0.1): 40, (10, 0.125): 40, (10, 0.15): 40, (10, 0.2): 3680, (10, 0.25): 40, (10, 0.3): 20000, (10, 0.35): 40, (10, 0.45): 40, (10, 0.5): 40, (12, 0.025): 40, (12, 0.0375): 40, (12, 0.05): 40, (12, 0.0625): 40, (12, 0.075): 40, (12, 0.0875): 40, (12, 0.1): 40, (12, 0.125): 40, (12, 0.15): 40, (12, 0.2): 3680, (12, 0.25): 40, (12, 0.3): 20000, (12, 0.35): 40, (12, 0.45): 40, (12, 0.5): 40, (14, 0.025): 40, (14, 0.0375): 40, (14, 0.05): 40, (14, 0.0625): 28, (14, 0.075): 28, (14, 0.0875): 0, (14, 0.1): 44, (14, 0.125): 44, (14, 0.15): 44, (14, 0.2): 0, (14, 0.25): 44, (14, 0.3): 20000, (14, 0.35): 44, (14, 0.45): 44, (14, 0.5): 44, (16, 0.025): 0, (16, 0.0375): 0, (16, 0.05): 0, (16, 0.0625): 0, (16, 0.075): 36, (16, 0.0875): 16, (16, 0.1): 24, (16, 0.125): 24, (16, 0.15): 24, (16, 0.2): 11840, (16, 0.25): 24, (16, 0.3): 20000, (16, 0.35): 20, (16, 0.45): 52, (16, 0.5): 28, (18, 0.025): 24, (18, 0.0375): 24, (18, 0.05): 24, (18, 0.0625): 28, (18, 0.075): 52, (18, 0.0875): 48, (18, 0.1): 28, (18, 0.125): 28, (18, 0.15): 28, (18, 0.2): 17980, (18, 0.25): 16, (18, 0.3): 20000, (18, 0.35): 20, (18, 0.45): 40, (18, 0.5): 44, (20, 0.025): 3620, (20, 0.0375): 4824, (20, 0.05): 2076, (20, 0.0625): 5764, (20, 0.075): 4812, (20, 0.0875): 9920, (20, 0.1): 5324, (20, 0.125): 2760, (20, 0.15): 2928, (20, 0.2): 20000, (20, 0.25): 9752, (20, 0.3): 20000, (20, 0.35): 11208, (20, 0.45): 11768, (20, 0.5): 8112}
parser = argparse.ArgumentParser(description='get some samples')
parser.add_argument('-ps', nargs='+', type=float)
parser.add_argument('-Ls', nargs='+', type=int)
parser = parser.parse_args()

if cluster == 'frontera':
    env_activate = "source /work2/08522/barratt/frontera/miniconda3/etc/profile.d/conda.sh && conda activate measurement_analysis && "
else:
    env_activate = ""

pLs = targets.keys()
ps = sorted(list(set([x[1] for x in pLs])))
Ls = sorted(list(set([x[0] for x in pLs])))
print(ps, Ls)

python_exe = "python "
python_file = "run.py"
string = ""
ps = ps if parser.ps is None else parser.ps
Ls = Ls if parser.Ls is None else parser.Ls
per_process = 90 # 9 batches per process - i.e. each process generates 10080 samples 
for L in Ls[::-1]:
    for p in ps:
        n_required = targets[L, p]
        N_batches_required = n_required // 56 # total number of batches required

        batches = [per_process] * (N_batches_required // per_process)
        if N_batches_required % per_process:
            batches = batches + [N_batches_required% per_process]
        for batch in batches:
            string = string + (env_activate+python_exe + python_file + f" {L} {p} {batch}\n")

print(string)
with open("param_sweep", "w") as f:
    f.write(string)
f.close()

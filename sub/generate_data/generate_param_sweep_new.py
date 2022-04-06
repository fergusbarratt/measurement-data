import argparse

parser = argparse.ArgumentParser(description='get some samples')
parser.add_argument('-ps', nargs='+', type=float)
parser.add_argument('-Ls', nargs='+', type=int)
parser.add_argument('-N_batches', nargs='?', default=90, type=int)
parser.add_argument('-dup', nargs='?', default=1, type=int)
parser = parser.parse_args()

env_activate = "source /work2/08522/barratt/frontera/miniconda3/etc/profile.d/conda.sh && conda activate measurement_analysis && "

python_exe = "python "
python_file = "run.py"
N_batches = parser.N_batches
dup = 2*parser.dup
string = ""
ps = [0.05, 0.075, 0.1, 0.125, 0.15, 0.25, 0.35, 0.45, 0.5] if parser.ps is None else parser.ps
for L in parser.Ls:
    for p in ps:
        string = string + (env_activate+python_exe + python_file + f" {L} {p} {N_batches}\n")*dup

with open("param_sweep_new", "w") as f:
    f.write(string)
f.close()

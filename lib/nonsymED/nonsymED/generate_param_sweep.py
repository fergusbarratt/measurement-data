env_activate = "source ~/miniconda3/etc/profile.d/conda.sh && conda activate haar_random && "
python_exe = "python "
python_file = "nonsymed.py"
N_batches = 90
dup = 2
string = ""
for L in [18, 20]:
    for p in [0.05, 0.075, 0.1, 0.125, 0.15, 0.25, 0.35, 0.45, 0.5]:
        string = string + (env_activate+python_exe + python_file + f" {L} {p} {N_batches}\n")*dup

print(string)
      
with open("param_sweep", "w") as f:
    f.write(string)
f.close()

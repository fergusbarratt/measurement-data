from itertools import product
from functools import reduce
Ls = [6, 8, 10, 12, 14, 16, 18, 20]
ps = [0.05, 0.075, 0.1, 0.125, 0.15, 0.25, 0.35, 0.45, 0.5]

pLs = sorted(list(product(Ls, range(len(ps)))))[::-1]

get_line = lambda p_ind, L: f"source /work2/08522/barratt/frontera/miniconda3/etc/profile.d/conda.sh && conda activate measurement_analysis && python decode.py {p_ind} {L}\n"

lines = reduce(lambda x, y: x+y, [get_line(p, l) for p, l in pLs])
print(lines)

with open('param_sweep', 'w') as f:
    f.write(lines)

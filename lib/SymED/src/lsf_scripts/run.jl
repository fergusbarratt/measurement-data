using NPZ, UUIDs
include("SymED.jl")

n_samples = 125
size = 14
T = 2*size
p = 0.05

data = SymED.n_evolve_sectors(n_samples, size, T, p)

id = string(uuid4())
npzwrite("data/$n_samples,$size,$p,$id.npy", data)

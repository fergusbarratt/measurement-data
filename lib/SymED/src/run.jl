using NPZ, UUIDs
include("SymED.jl")

n_samples = 1#56
size = 10
T = 2*size
p = 0.5

data = SymED.n_evolve_sectors(n_samples, size, T, p, threading=false)

id = string(uuid4())
npzwrite("data/$n_samples,$size,$p,$id.npy", data)

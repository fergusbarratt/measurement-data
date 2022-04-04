using NPZ, UUIDs
using LinearAlgebra
BLAS.set_num_threads(1)

include("../SymED.jl")

n_samples = 1
size = 12
T = 2*size
p = 0.5
threading=false

println("getting $n_samples sample(s), size $size, p=$p, threading=$threading, with $(Threads.nthreads()) threads")

data = SymED.n_evolve_sectors(n_samples, size, T, p, threading=false)

id = string(uuid4())
npzwrite("test_data/$n_samples,$size,$p,$id.npy", data)

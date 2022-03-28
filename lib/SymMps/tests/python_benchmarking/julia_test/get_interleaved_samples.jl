using Distributed

cores_per_node = 8

num_threads = Threads.nthreads()
num_procs = cores_per_node ÷ num_threads
batch_size = cores_per_node
# each node should produce cores_per_node results in each batch
# num threads is specified, and num_procs makes up the batch size

addprocs(num_procs) # num threads=28, 56 cores-> 2 processes

p = parse(Float64, ARGS[1])
L = parse(Int64, ARGS[2])
n_batches = parse(Int64, ARGS[3])

println("threads: ", num_threads, ", processes: ", num_procs, ", params: ", p, ", ", L, ", ", n_batches)

@everywhere begin
    # each of these runs multithreaded on $JULIA_NUM_THREADS
    include("interleaved.jl")


    # num_procs of these are run at a time, 
    # each uses num_threads to gets batch size samples
    function get_batch(L, p, batch_size)
        Interleaved.runmany(batch_size, L, 4*L, p, "data")
    end
end

pmap(a -> get_batch(L, p, batch_size), 1:n_batches)

using Distributed

cores_per_node = Threads.nthreads()


threads_per_process = 2
num_threads = cores_per_node ÷ threads_per_process # total number of threads. 
num_procs_add = (cores_per_node ÷ num_threads) # number of TOTAL processes to create

# how many samples to get in each threaded run.
# should divide num_threads: i.e. if num_threads = 28, and batch_size = 56, then 
# each thread will get a queue of 2 jobs per batch. 
batch_size = cores_per_node

addprocs(num_procs_add) # num threads=28, 56 cores-> 2 processes

p = parse(Float64, ARGS[1])
L = parse(Int64, ARGS[2])
n_batches = parse(Int64, ARGS[3])

println("threads: ", num_threads, ", new processes: ", num_procs_add, ", params: ", p, ", ", L, ", ", n_batches)

@everywhere begin
    # each of these runs multithreaded on $JULIA_NUM_THREADS
    include("interleaved.jl")


    # num_procs_add of these are run at a time, 
    # each uses num_threads to gets batch size samples
    function get_batch(L, p, batch_size)
        Interleaved.runmany(batch_size, L, 4*L, p, "data")
    end
end

pmap(a -> get_batch(L, p, batch_size), 1:n_batches)

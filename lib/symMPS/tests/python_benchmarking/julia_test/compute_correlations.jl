using Distributed

cores_per_node = 8

num_threads = Threads.nthreads()
num_procs = cores_per_node ÷ num_threads
batch_size = cores_per_node
# each node should produce cores_per_node results in each batch
# num threads is specified, and num_procs makes up the batch size

addprocs(num_procs) # num threads=28, 56 cores-> 2 processes

println("processes: ", num_procs, ", threads: ", num_threads)
@everywhere begin
    using SymMps
    using BSON

    data_loc = "data/"
    Z = [1 0; 0 -1]
    corr_dir = data_loc*"correlations/"
    data_loc = data_loc*"states/" # assume the states are in a subfolder
    mkpath(corr_dir)

    function readstatedata(data_loc=data_loc)  
        for datapath = readdir(data_loc) # for each directory containing states (use processes?)
            try
                N, t_max, pₘ = split(datapath, ',')
                N, t_max = map(a->parse(Int64, a), [N, t_max])
                len = length(readdir(data_loc*datapath)) # number of states
                if len != 1 && !isfile(corr_dir*datapath*".bson")
                    println()
                    println(datapath)
                    CCs = zeros(len, N, N)
                    zs = zeros(len, 1, N)
                    Threads.@threads for (i, statepath) = collect(enumerate(readdir(data_loc*datapath))) # for each state (use threads?)
                        # get the state
                        A = SymMps.load(data_loc*datapath*'/'*statepath)

                        # compute the quantities
                        try
                            CC = SymMps.expectation2site!(A, kron(Z, Z), false)
                            z = SymMps.expectation1site!(A, Z)

                            # write the quantities
                            CCs[i, :, :] = CC
                            zs[i, :, :] = z
                        catch y
                            println(y)
                        end

                        print("$i / $len      \r")
                    end
                    bson(corr_dir*datapath*".bson", correlations=CCs, zs=zs);
                    println()
                end
                print('.')
            catch y
                println(y)
            end
        end
    end
end

readstatedata(data_loc)

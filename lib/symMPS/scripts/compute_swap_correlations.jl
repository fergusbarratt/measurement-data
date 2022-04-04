using Revise
using Distributed
using SymMps
using BSON

cores_per_node = Threads.nthreads()


num_procs = 1
num_threads = cores_per_node ÷ num_procs # total number of threads. 

# how many samples to get in each threaded run.
# should divide num_threads: i.e. if num_threads = 28, and batch_size = 56, then 
# each thread will get a queue of 2 jobs per batch. 
batch_size = cores_per_node

addprocs(num_procs) # num threads=28, 56 cores-> 2 processes


println("processes: ", num_procs, ", threads: ", num_threads)

data_loc = "data/"
data = zeros(2,2,2,2)
data[1,1,1,1] = data[2,2,2,2] = data[1,2,2,1] = data[2,1,1,2] = 1
SWAP = reshape(data, (4, 4))

corr_dir = data_loc*"correlations/"
data_loc = data_loc*"states/" # assume the states are in a subfolder
mkpath(corr_dir)

function readstatedata(data_loc=data_loc)  
    # for each directory containing multiple states (use processes?)
    for datapath = readdir(data_loc)
            # get the information from the filename
            N, t_max, pₘ = split(datapath, ',')
            N, t_max = map(a->parse(Int64, a), [N, t_max])

            # How many states in the folder
            len = length(readdir(data_loc*datapath))
            # If there are states in the folder, 
            # and the folder hasn't already been processed to bson
            if len != 1 && !isfile(corr_dir*datapath*".bson")
                println()
                println(datapath)
                # initialize where the different bits of the correlators will go
                oswaps = zeros(len, 1, N-1)
                eswaps = zeros(len, 1, N-1)
                oswapswaps = zeros(len, N-1, N-1)
                eswapswaps = zeros(len, N-1, N-1)

                # get all the state paths (odd and even)
                a = readdir(data_loc*datapath)
                # get just the state paths that start with 'e' (i.e. the even ones)
                statepaths = filter(a-> a[1] == 'e', a)

                Threads.@threads for (i, statepath) = collect(enumerate(statepaths)) # for each state (use threads?)
                    # add the prefix
                    even_path = data_loc*datapath*'/'*statepath
                    # get the state
                    Aₑ = SymMps.load(even_path)

                    # get the odd path from the even one
                    odd_statepath = replace(statepath, "e"=>"o", count=1)
                    odd_path = data_loc*datapath*'/'*odd_statepath

                    # get the corresponding odd state
                    Aₒ = SymMps.load(odd_path)

                    # compute the quantities
                    # 2nd part
                    swapₑ = SymMps.localexpectation2site!(Aₑ, SWAP)
                    swapₒ = SymMps.localexpectation2site!(Aₒ, SWAP)

                    # first part
                    swapswapₑ = SymMps.sym_twosite_C(Aₑ, SymMps.SWAP)
                    swapswapₒ = SymMps.sym_twosite_C(Aₒ, SymMps.SWAP)

                    # write the quantities to the preallocated arrays
                    #  2nd part
                    oswaps[i, :, :] = swapₒ
                    eswaps[i, :, :] = swapₑ

                    # 1st part
                    oswapswaps[i, :, :] = swapswapₒ
                    eswapswaps[i, :, :] = swapswapₑ

                    print("$i / $len      \r")
                end
                # save the results to a bson file. 
                bson(corr_dir*datapath*".bson", eswaps=eswaps, oswaps=oswaps, eswapswaps=eswapswaps, oswapswaps=oswapswaps);
                println()
            end
            print('.')
    end
end

readstatedata(data_loc)

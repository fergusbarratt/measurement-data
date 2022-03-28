module Interleaved
    using SymMps, TensorKit, UUIDs, BSON
    """ 
    Sweep left and right and back again, appying born singlet/triplet measurements 
     or random unitaries 
    """
    function TEBDmeasuresweep!(A, p; dimerization=0, Un=SymMps.randomunitary, P₀=SymMps.P₀, P₁=SymMps.P₁, verbose = true, truncdim = 200, η = 1e-6)
        verbose && println("sweep")
        N = length(A)
        pₑ = minimum([p+dimerization, 1.])
        pₒ = maximum([p-dimerization, 0.])

        # Assume a right canonical MPS, so the 'center' tensor is A[1] (remember 1 indexing)
        AC = A[1]
        rank = 1
        for k = 1:N-2
            AC = AC/norm(AC) # normalise AC (normalise the state)
            @tensor AAC[-1,-2,-3,-4] := AC[-1,-2,1]*A[k+1][1,-3,-4] # join two sites together
            
            if isodd(k) # update AAC if index is odd, else just shift the orth centre
                if rand() < pₑ
                    # if we're measuring, get the ev of P₀
                    @tensor p_born = AAC[a, b', c', d]*P₀[b', c', b, c]*conj(AAC[a, b, c, d])
                    if rand() < real(p_born)
                        @tensor AAC[a, b, c, d] := AAC[a, b', c', d]*P₀[b', c', b, c] # U_: V->V'
                    else
                        @tensor AAC[a, b, c, d] := AAC[a, b', c', d]*P₁[b', c', b, c] # U_: V->V'
                    end
                else
                    U_ = Un()
                    @tensor AAC[a, b, c, d] = AAC[a, b', c', d]*U_[b', c', b, c] # U_: V->V'
                end
            end
            verbose && print("Sweep L2R: apply site $(k:k+1) $(rank)       \r")

            AL, S, V = tsvd(AAC, (1, 2), (3, 4), trunc=truncerr(η))
            rank = max(rank, dim(domain(S)))
            A[k] = AL 
            @tensor AC[a, b, c] := S[a, a']*V[a', b, c]
        end
        AC = AC/norm(AC) # normalise AC (normalise the state)

        k = N-1
        @tensor AAC[-1,-2,-3,-4] := AC[-1,-2,1]*A[k+1][1,-3,-4]
        if isodd(k) # update AAC if index is odd, else just shift the orth centre
            if rand() < pₑ
                @tensor p_born = AAC[a, b', c', d]*P₀[b', c', b, c]*conj(AAC[a, b, c, d])
                if rand() < real(p_born)
                    @tensor B[a, b, c, d] := AAC[a, b', c', d]*P₀[b', c', b, c] # U_: V->V'
                    AAC = B
                else
                    @tensor B[a, b, c, d] := AAC[a, b', c', d]*P₁[b', c', b, c] # U_: V->V'
                    AAC = B
                end
            else
                U_ = Un()
                @tensor B[a, b, c, d] := AAC[a, b', c', d]*U_[b', c', b, c] # U_: V->V'
                AAC = B
            end
        end
        verbose && print("Sweep L2R: apply site $(k:k+1) $(rank)       \r")
        verbose && println()
        for k = N-1:-1:2
            U, S, AR = tsvd(AAC, (1, 2), (3, 4), trunc=truncerr(η))
            rank = max(rank, dim(domain(S)))

            A[k+1] = AR 
            @tensor AC[a, b, c] := U[a, b, c']*S[c', c]
            AC = AC/norm(AC) # normalise AC (normalise the state)
            @tensor AAC[:] := A[k-1][-1,-2,1] * AC[1,-3,-4]
            if isodd(k) # update AAC if index is odd, else just shift the orth centre
                if rand() < pₒ
                    @tensor p_born = AAC[a, b', c', d]*P₀[b', c', b, c]*conj(AAC[a, b, c, d])
                    if rand() < real(p_born)
                        @tensor AAC[a, b, c, d] := AAC[a, b', c', d]*P₀[b', c', b, c] # U_: V->V'
                    else
                        @tensor AAC[a, b, c, d] := AAC[a, b', c', d]*P₁[b', c', b, c] # U_: V->V'
                    end
                else
                    U_ = Un()
                    @tensor AAC[a, b, c, d] = AAC[a, b', c', d]*U_[b', c', b, c] # U_: V->V'
                end
            end
            verbose && print("Sweep R2L: apply site $(k:k+1) $(rank)       \r")
        end

        verbose && println()
        k = 1
        U, S, AR = tsvd(AAC, (1, 2), (3, 4), trunc=truncerr(η))

        A[k+1] = AR
        @tensor AC[a, b, c] := U[a, b, c']*S[c', c]
        A[1] = AC
        return A
    end


    function run(N, t_max, pₘ, γ=0, verbose=false, η=1e-6)
        #script
        A = SymMps.singletmps(N) # MPS of tiled singlets. 
        SymMps.e1!(A, [1. 0.; 0. 1.]) # right orthonormalize

        allees = Array{Float64}(undef, N-1, t_max)
        for t = 1:t_max
            verbose && println("Time $t")
            allees[:, t] = real(SymMps.entanglemententropy!(A)) # get the entanglement entropy
            TEBDmeasuresweep!(A, pₘ, dimerization=γ, verbose=verbose, η = η);
            verbose && println()
        end
        return A, allees
    end

    function runmany(n_runs, N, t_max, pₘ, dirname="data", proc_id="", verbose=true, η = 1e-6; γ=0)
        # this is where the data goes
        out = Array{Float64}(undef, n_runs, N-1, t_max)

        # saving dir
        key = string(N)*","*string(t_max)*","*string(pₘ)

        dir = dirname*"/"*"states"*"/"*key*"/" # data/states/key/...
        verbose && println("saving states in ", dir)
        mkpath(dir)

        # run in many threads
        Threads.@threads for n = 1:n_runs
            # do the run
            A, out[n, :, :] = run(N, t_max, pₘ, γ, verbose, η)

            # save the run.
            SymMps.save(A, dir)
        end

        folder = dirname*"/"*"entropy/" # data/entropy/key.bson
        verbose && println("saving entropy data in ", folder)
        # save the entropy data
        mkpath(folder)
        bson(folder*key*","*string(uuid4())*".bson", entropy=out);
    end
end

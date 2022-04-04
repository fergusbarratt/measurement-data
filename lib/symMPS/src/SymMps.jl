module SymMps
    using Revise
    using TensorKit, LinearAlgebra, UnicodePlots, ProgressMeter, UUIDs, BSON, Statistics
    """
    get expectation at each site during a sweep left, then back
    To canonicalise, do e1!(A, I)
    """
    function expectation1site!(A, O, η = 1e-9)
        N = length(A)
        expval = Vector{ComplexF64}(undef, N)
        # input MPS has no particular gauge. 
        # shift the gauge to the far right - right orthonormalise the MPS. 
        for k = 1:N
            # sweep right
            AL, S, V = tsvd(A[k], (1, 2), (3,), trunc=truncerr(η))
            A[k] = AL
            if k < N
                SV = S*V
                @tensor B[a, b, c] := SV[a, a']*A[k+1][a', b, c]
                A[k+1] = B
            end
        end
        for k = N:-1:1
            # sweep back 
            # at each instance here, A[k] is in centre gauge. 
            # so this is the reduced density matrix. 
            @tensor ρ[a, b] := A[k][α, a, β]*conj(A[k][α, b, β])

            # this is small, so efficient. 
            # This way we can take evs of non-symmetric operators. 
            ρ = convert(Array, ρ)
            expval[k] = tr(ρ*O)

            # now shift the gauge left
            U, S, AR = tsvd(A[k], (1,), (2, 3), trunc=truncerr(η))
            A[k] = AR
            if k > 1
                US = U*S
                @tensor B[a, b, c] := A[k-1][a, b, c']*US[c', c]
                A[k-1] = B
            end
        end
        return expval
    end
    e1! = expectation1site!
    orth! = a->expectation1site!(a, I)
    """
    Apply operator to unitary
    Args: 
        A: MPS  
        U: symmetric operator (four site tensor)
        k: site to apply the operator 
    """
    function apply!(A, U, k; η=0)
        @tensor AA[a, b, c, d] := U[b', c', b, c]*A[k][a, b', d']*A[k+1][d', c', d] 
        AL, S, AR = tsvd(AA, (1, 2), (3, 4), trunc=truncerr(η)) # QR decomp
        @tensor AL[a, b, c] := AL[a, b, c']*S[c', c]

        A[k] = AL
        A[k+1] = AR

        return A
    end

    """
    Symmetric two site Cᵢⱼ
    """
    function sym_twosite_C(A, U; η=0)
        N = length(A)
        C = ones(Float64, N-1, N-1)
        for i in 1:N-1
            for j in 1:i-1
                B = deepcopy(A)
                apply!(B, U, i)
                apply!(B, U, j)
                C[i, j] = real(SU2.overlap(A, B))
                C[j, i] = C[i, j]
            end
        end
        return C
    end


    """
    Get local expectation values of 2 site operator (i.e. not full matrix)
    """
    function localexpectation2site!(A, O, verbose=false, η=0)
        N = length(A)
        es = zeros(ComplexF64, N-1)
        # input MPS has no particular gauge. 
        # shift the gauge to the far right - right orthonormalise the MPS. 
        for k = 1:N
            # sweep right
            AL, S, V = tsvd(A[k], (1, 2), (3,), trunc=truncerr(η))
            A[k] = AL
            if k < N
                SV = S*V
                @tensor B[a, b, c] := SV[a, a']*A[k+1][a', b, c]
                A[k+1] = B
            end
        end
        for k = N:-1:1

            if k >= 2
                # get ρ[k-1, k]
                @tensor AAC[a, b, c, d] := A[k-1][a, b, c']*A[k][c', c, d] # two site 
                @tensor ρ[a, b, c, d] := AAC[a', a, b, c']*conj(AAC[a', c, d, c'])

                ρ = convert(Array, ρ)

                verbose && print("getting ($(k-1), $k)")

                es[k-1] = tr(reshape(ρ, (4, 4))*O)
            end
                

            # now shift the gauge left
            U, S, AR = tsvd(A[k], (1,), (2, 3), trunc=truncerr(η))
            A[k] = AR
            if k > 1
                US = U*S
                @tensor B[a, b, c] := A[k-1][a, b, c']*US[c', c]
                A[k-1] = B
            end
        end
        verbose && println()
        return real.(es)
    end
    les2! = localexpectation2site!


    """entropy of an array"""
    function getentropy(S)
        sum(-S.^2 .* log2.(S.^2 .+ 1e-15))
    end

    """
    Get the entanglement entropy of a matrix product state (at every site)
    """
    function entanglemententropy!(A)
        N = length(A)
        entropy = Vector{ComplexF64}(undef, N-1)
        # input MPS has no particular gauge. 
        # shift the gauge to the far right - right orthonormalise the MPS. 
        for k = 1:N
            # sweep right
            AL, S, V = tsvd(A[k], (1, 2), (3,), trunc=notrunc())
            A[k] = AL
            
            if k < N
                # get the entanglement entropy from the schmidt coefficients, blockwise
                ee = 0.
                for sectorblock = blocks(S)
                    sector, block = sectorblock
                    ee += dim(sector)*getentropy(diag(block))
                end
                entropy[k] = ee
                #assert(getentropy(diag(convert(Array, S))) ≈ ee) # assuming this. 
                
                SV = S*V
                @tensor B[a, b, c] := SV[a, a']*A[k+1][a', b, c]
                A[k+1] = B
            end
        end
        for k = N:-1:1
            # sweep back 
            # at each instance here, A[k] is in centre gauge. 
            # so this is the reduced density matrix. 
            #@tensor ρ[a, b] := A[k][α, a, β]*conj(A[k][α, b, β])

            # this is small, so efficient. 
            # This way we can take evs of non-symmetric operators. 
            #ρ = convert(Array, ρ)
            #expval[k] = tr(ρ*O)

            # now shift the gauge left
            U, S, AR = tsvd(A[k], (1,), (2, 3), trunc=notrunc())
            A[k] = AR
            if k > 1
                US = U*S
                @tensor B[a, b, c] := A[k-1][a, b, c']*US[c', c]
                A[k-1] = B
            end
        end
        return entropy
    end

    function save(A, dir="./", tag="", id="")
        dicts = map(a -> convert(Dict, a), A) # convert the tensors to dicts
        id = id=="" ? string(uuid4()) : id
        filename = dir*tag*","*id*".bson"
        bson(filename, A=dicts) # save the bson
        id
    end

    function load(filename)
        dicts = BSON.load(filename)[:A] # get the array of dicts out of the bson
        A = map(a-> convert(TensorMap, a), dicts) # convert the dicts to tensors
    end

    function readentropydata(data_loc="./entropy/")  
        for datapath = readdir(data_loc)
            n_runs, N, t_max, pₘ, pᵤ = split(splitext(datapath)[1], ',')
            n_runs, N, t_max = map(a->parse(Int64, a), [n_runs, N, t_max])
            pₘ, pᵤ = map(a -> parse(Float64, a), [pₘ, pᵤ])

            data = BSON.load(data_loc*datapath)[:entropy]
            av_entropy = mean(data, dims=1)
            #plt = lineplot(1:size(av_entropy, 2), vec(av_entropy[:, :, end]))
            #show(plt)
            plt = lineplot(1: size(av_entropy, 3), vec(av_entropy[:, size(av_entropy, 2) ÷ 2, :]), xscale=:log10)
            show(plt)
            println()
        end
    end

    module SU2
        using TensorKit, LinearAlgebra, UnicodePlots, ProgressMeter, UUIDs, BSON, Statistics
        """
        A set of useful SU(2) symmetric objects
        """
        𝕍₀ = SU₂Space(0=>1) # dummy space
        𝕍ₚ = SU₂Space(1/2=>1) # physical space
        P₀ = TensorMap(randisometry, 𝕍ₚ ⊗ 𝕍ₚ, SU₂Space(0=>1))
        P₀ = P₀ * P₀' # (a) projector onto rhe spin 0 subspace
        P₁ = TensorMap(randisometry, 𝕍ₚ ⊗ 𝕍ₚ, SU₂Space(1=>1))
        P₁ = P₁ * P₁' # (a) projector onto the spin 1 subspace
        data = zeros(2,2,2,2)
        data[1,1,1,1] = data[2,2,2,2] = data[1,2,2,1] = data[2,1,1,2] = 1
        SWAP = TensorMap(data, 𝕍ₚ ⊗ 𝕍ₚ, 𝕍ₚ ⊗ 𝕍ₚ) # SWAP gate

        """
        Get a random SU(2) symmetric unitary
        """
        function randomunitary()
            return TensorMap(randisometry, ComplexF64, 𝕍ₚ ⊗ 𝕍ₚ, 𝕍ₚ ⊗ 𝕍ₚ)
        end

        """
        Create an MPS of tiled singlets (not canonical)
        """
        function singletmps(N)
            A = Vector{Any}(undef, N)
            for n = 1:N
                if isodd(n)
                    # a singlet (two sites)
                    singlet = TensorMap(ones, ComplexF64, 𝕍₀, 𝕍ₚ ⊗ 𝕍ₚ ⊗ 𝕍₀)
                    # split the singlet
                    Q, R = leftorth(singlet, (1, 2), (3, 4))
                    A[n] = Q
                    A[n+1] = R
                end
            end
            return A
        end

        """
        create a random bond dimension 2 symmetric mps. 
        """
        aspace = SU₂Space(0=>1, 1/2=>1)
        function rand2mps(N; 𝕍ᵥ=aspace)
            A = Vector{Any}(undef, N)
            A[1] = TensorMap(ones, ComplexF64, 𝕍₀,  𝕍ₚ ⊗ 𝕍ᵥ)
            for n = 2:N
                A[n] = TensorMap(ones, ComplexF64, 𝕍ᵥ,  𝕍ₚ ⊗ 𝕍ᵥ)
            end
            A[N] = TensorMap(ones, ComplexF64, 𝕍ᵥ,  𝕍ₚ ⊗ 𝕍₀)
            return A
        end

        """
        Overlap between two MPS
        """
        function overlap(A, B)
            l = Tensor(ones, 𝕍₀' ⊗ 𝕍₀)
            for (Ai, Bi) in zip(A, B)
                @tensor l[a, b] := l[a', b']*Ai[a', c', a]*conj(Bi[b', c', b])
            end
            tr(convert(Array, l))
        end

        """
        get ⟨O(i, j)⟩ for all i, j
        """
        function expectation2site!(A, O, verbose=true, η = 0)
            N = length(A)
            es = zeros(ComplexF64, N, N)
            # input MPS has no particular gauge. 
            # shift the gauge to the far right - right orthonormalise the MPS. 
            for k = 1:N
                # sweep right
                AL, S, V = tsvd(A[k], (1, 2), (3,), trunc=truncerr(η))
                A[k] = AL
                if k < N
                    SV = S*V
                    @tensor B[a, b, c] := SV[a, a']*A[k+1][a', b, c]
                    A[k+1] = B
                end
            end
            for k = N:-1:1
                new_A = deepcopy(A)
                for j = k-1:-1:1 # order is [[(N, N-1), (N, N-2),...], [(N-1, N-2), (N-1, N-3)..], ...
                    # take new_A, and sweep left, applying SWAPs, and appending the density matrix 
                    # at (k-1, k) (which is actually (j, k), post swap) to ρs

                    # get ρ'[k-1, k] = ρ[j, k]
                    @tensor AAC[a, b, c, d] := new_A[j][a, b, c']*new_A[j+1][c', c, d] # two site 
                    @tensor ρ[a, b, c, d] := AAC[a', a, b, c']*conj(AAC[a', c, d, c'])

                    ρ = convert(Array, ρ)

                    verbose && print("writing $j, $k, then swapping $j, $(j+1)             \r")
                    es[j, k] = tr(reshape(ρ, (4, 4))*O)
                    if j > 1
                        @tensor AAC[a, b, c, d] := SWAP[b', c', b, c]*new_A[j][a, b', d']*new_A[j+1][d', c', d] 
                        AL, S, AR = tsvd(AAC, (1, 2), (3, 4), trunc=truncerr(η)) # QR decomp
                        #@tensor AR[a, b, c] := S[a, a']*AR[a', b, c]
                        @tensor AL[a, b, c] := AL[a, b, c']*S[c', c]

                        new_A[j] = AL
                        new_A[j+1] = AR
                    end
                end
                    

                # now shift the gauge left
                U, S, AR = tsvd(A[k], (1,), (2, 3), trunc=truncerr(η))
                A[k] = AR
                if k > 1
                    US = U*S
                    @tensor B[a, b, c] := A[k-1][a, b, c']*US[c', c]
                    A[k-1] = B
                end
            end
            verbose && println()
            return real.(es+es')
        end
        e2! = expectation2site!
    end
    module U1
        using TensorKit, LinearAlgebra, UnicodePlots, ProgressMeter, UUIDs, BSON, Statistics
        """
        A set of useful U(1) symmetric objects
        """
        𝕍₀ = U₁Space(0=>1) # dummy space
        𝕍ₚ = U₁Space(1/2=>1) # physical space
        P₀ = TensorMap(randisometry, 𝕍ₚ ⊗ 𝕍ₚ, U₁Space(0=>1))
        P₀ = P₀ * P₀' # (a) projector onto rhe spin 0 subspace
        P₁ = TensorMap(randisometry, 𝕍ₚ ⊗ 𝕍ₚ, U₁Space(1=>1))
        P₁ = P₁ * P₁' # (a) projector onto the spin 1 subspace
    end
end # module

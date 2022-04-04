module SymED
    using TensorKit, StatsBase

    𝕍 = U₁Space(-1/2=>1, 1/2=>1)
    up = U₁Space(1/2=>1)
    down = U₁Space(-1/2=>1)
    P₋ = TensorMap(ones, ComplexF64, 𝕍, down)
    P₊ = TensorMap(ones, ComplexF64, 𝕍, up)

    P₋ = P₋*P₋'
    P₊ = P₊*P₊'
    
    function initial_state(s, N)
        𝕍ₛ = U₁Space(s=>1)
        𝕍ₜ = reduce(⊗, [𝕍 for _ in 1:N])
        return Tensor(ones, ComplexF64, 𝕍ₜ ⊗ 𝕍ₛ)/sqrt(binomial(N, N÷2+s))
    end

    function random_unitary()
        return TensorMap(randisometry, ComplexF64, 𝕍 ⊗ 𝕍, 𝕍 ⊗ 𝕍)
    end

    function apply_unitary(state, U, i)
        N = numout(state)-1 # all state indices but one, which holds the charge
        ncon_strings = ([-i, -(i+1), i, (i+1)], [(j==i || j==i+1) ? j : -j  for j=1:N+1])
        @ncon((U, state), ncon_strings)
    end

    function apply_operator(state, O, i)
        N = numout(state)-1
        ncon_strings = ([-i, i], [j==i ? i : -j  for j=1:N+1])
        @ncon((O, state), ncon_strings)
    end

    # gets the vector representation of the block. 
    # i.e. coefficients of each bitstring in the block. 
    function vec_from_state(state)
        first(blocks(state))[2] 
    end

    """
    Get the expectation of Z (without allocating 
    """
    function p₀(state, i)
        app_state = apply_operator(deepcopy(state), P₋, i)
        return real.(vec_from_state(state)'*vec_from_state(app_state))[1]
    end

    """
    Crazy allocations here (i think)
    """
    function ρ(state, i)
        N = numout(state)-1 # all state indices but one, which holds the charge
        ncon_strings = ([(j!=i) ? j : -1  for j=1:N+1], [(j!=i) ? j : -2  for j=1:N+1])
        @ncon((state, state'), ncon_strings)
    end

    """
    Crazy allocations here (i think)
    """
    function norm2(state)
        N = numout(state)-1 # all state indices but one, which holds the charge
        ncon_strings = ([j  for j=1:N+1], [j  for j=1:N+1])
        t = @ncon((state, state'), ncon_strings)
        sqrt.(real.(block(t, first(blocksectors(t)))))[1]
    end

    """
    Should be fewer allocations here hopefully
    """
    function norm(state)
        v = vec_from_state(state)
        return sqrt.(real.(v' * v))[1]
    end

    function measure(state, i)
        #down = rand() < real(convert(Array, SymED.ρ(state, i))[1, 1])
        down = rand() < 1-p₀(state, i)
        if !down
            state = apply_operator(state, P₋, i)
        else
            state = apply_operator(state, P₊, i)
        end
        (state/norm(state), 2*iszero(down)-1)
    end

    function evolve_sector(N, T, p, Q; verbose=true)
        ψ = initial_state(Q, N)
        measure_frame = sample(0:1, Weights([1-p, p]), (N, 2*T+1))
        for t in 1:T-1
            ## Even
            # evolve
            verbose && print("$(t+1) / $T  \r")
            for i in 1:2:N-1
                ψ = apply_unitary(ψ, random_unitary(), i)
            end

            # measure
            for i in 1:N 
                if measure_frame[i, 2*t] == 1
                    ψ, out = measure(ψ, i)
                    measure_frame[i, 2*t] = out
                end
            end

            ## Odd
            # evolve
            for j in 2:2:N-1
                ψ = apply_unitary(ψ, random_unitary(), j)
            end

            # measure
            for j in 1:N 
                if measure_frame[j, 2*t+1] == 1
                    ψ, out = measure(ψ, j)
                    measure_frame[j, 2*t+1] = out
                end
            end
        end
        ψ₂ = deepcopy(ψ)
        for j in 1:N 
            # measure all the qubits at the end
            ψ₂, out = measure(ψ₂, j)
            measure_frame[j, 2*T+1] = out 
        end
        verbose && println()
        ψ, measure_frame
    end

    function n_evolve_sectors(n_samples, N, T, p; verbose=true, threading=true)
        records = zeros(Int, n_samples, 2, N, 2*T+1)
        Q = 0
        verbose && println("Q")
        if threading
            Threads.@threads for n in 1:n_samples
                verbose && print("$n\n ")
                records[n, 1, :, :] .= evolve_sector(N, T, p, Q)[2]
            end
        else
            for n in 1:n_samples
                verbose && print("$n\n ")
                records[n, 1, :, :] .= evolve_sector(N, T, p, Q)[2]
            end
        end

        Q = 1
        verbose && println("Q+1")
        if threading
            Threads.@threads for n in 1:n_samples
                verbose && print("$n\n ")
                records[n, 2, :, :] .= evolve_sector(N, T, p, Q)[2]
            end
        else
            for n in 1:n_samples
                verbose && print("$n\n ")
                records[n, 2, :, :] .= evolve_sector(N, T, p, Q)[2]
            end
        end

        verbose && println()
        verbose && flush(stdout)
        records
    end
end

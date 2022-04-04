using BenchmarkTools
using TensorKit
include("SymED.jl");

ψ = SymED.initial_state(0, 14);
U = SymED.random_unitary();

#@benchmark include("run.jl")
@btime SymED.apply_unitary(ψ, U, 1) seconds=3;

f(ψ) = @tensor ψ[:] = U[-1, -2, 1, 2]*ψ[1, 2, -3, -4, -5, -6, -7, -8, -9, -10, -11, -12, -13, -14, -15];
@btime f(ψ) seconds=3;

@btime SymED.apply_operator(ψ, SymED.P₋, 1) seconds=3;
#@allocated

using Test
using SymMps
using LinearAlgebra
include("interleaved.jl")

Z = [1 0; 0 -1]
X = [0 1; 1 0]
Y = [0 -im; im 0]
SWAP = [1 0 0 0; 0 0 1 0; 0 1 0 0; 0 0 0 1]

Test.@testset "interleaved" begin
    # evs in a singletmps
    A = SymMps.singletmps(8)
    Interleaved.TEBDmeasuresweep!(A, 0.)
    Interleaved.TEBDmeasuresweep!(A, 1.0)
    println(SymMps.les2!(A, SWAP))
end

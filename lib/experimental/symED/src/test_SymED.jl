using Test, LinearAlgebra, Revise, TensorKit
include("SymED.jl")

Test.@testset "norms" begin
    ψ = SymED.initial_state(0, 6)     
    Test.@test SymED.norm(ψ) ≈ 1.
    Test.@test SymED.norm2(ψ) ≈ 1.
    ρ = convert(Array, SymED.ρ(ψ, 3))
    Test.@test tr(ρ) ≈ 1

    ψ = SymED.apply_unitary(ψ, SymED.random_unitary(), 2)
    Test.@test SymED.norm(ψ) ≈ 1.
    Test.@test SymED.norm2(ψ) ≈ 1.
    ρ = convert(Array, SymED.ρ(ψ, 3))
    Test.@test tr(ρ) ≈ 1
end

Test.@testset "probs" begin
    ψ = SymED.initial_state(0, 6)     
    zz = convert(Array, SymED.ρ(ψ, 1))[1, 1]
    zz_ = 1-SymED.p₀(ψ, 1)
    Test.@test zz ≈ 0.5
    Test.@test zz_ ≈ 0.5

    for i in -3:3
        ψ = SymED.initial_state(i, 6)     
        zz_ = 1-SymED.p₀(ψ, 1)
        zz = convert(Array, SymED.ρ(ψ, 1))[1, 1]
        Test.@test zz ≈ zz_
    end
end

using Test
using SymMps
using LinearAlgebra

Z = [1 0; 0 -1]
X = [0 1; 1 0]
Y = [0 -im; im 0]

Test.@testset "EVS" begin
    B = SymMps.SU2.rand2mps(4)


    #normalising twice gets the same thing
    out1 = SymMps.e1!(B, I)
    Test.@test out1 ≈ ones(size(out1))
    out1 = SymMps.e1!(B, I)
    Test.@test out1 ≈ ones(size(out1))

    #eving twice gets the same thing
    out1 = SymMps.e1!(B, Z)
    out2 = SymMps.e1!(B, Z)
    Test.@test out1 ≈ out2

    out1 = SymMps.SU2.e2!(B, kron(Z, Z))
    out2 = SymMps.SU2.e2!(B, kron(Z, Z))
    Test.@test out2 ≈ out2

    out1 = SymMps.les2!(B, kron(Z, Z))
    out2 = SymMps.les2!(B, kron(Z, Z))
    Test.@test out2 ≈ out2

    # evs in a singletmps
    A = SymMps.SU2.singletmps(4)
    v = diagm(1=>[-1, 0, -1])
    v = v+v'
    out1 = SymMps.SU2.e2!(A, kron(Z, Z))
    out2 = SymMps.SU2.e2!(A, kron(Y, Y))
    out3 = SymMps.SU2.e2!(A, kron(X, X))
    Test.@test v ≈ out1
    Test.@test v ≈ out2
    Test.@test v ≈ out3

    # test zz ev same as e2!
    out4 = SymMps.les2!(A, kron(Z, Z))
    Test.@test v ≈ diagm(1=>out4, -1=>out4)

    # test local ev
    out5 = SymMps.les2!(A, kron(Z, Matrix(I, 2, 2)))
    comp = zeros(size(out5))
    Test.@test out5 ≈ comp

    # test local ev
    out6 = SymMps.les2!(A, kron(X, Matrix(I, 2, 2)))
    comp = zeros(size(out6))
    Test.@test out6 ≈ comp

    # test swap ev
    out7 = SymMps.les2!(A, reshape(convert(Array, SymMps.SU2.SWAP), (4, 4)))
    Test.@test out7 ≈ [-1.0, 0.5, -1.0]

    # Test e2!
    A = SymMps.SU2.singletmps(6)
    v = diagm(1=>[-1, 0, -1, 0, -1])
    v = v+v'
    out = SymMps.SU2.e2!(A, kron(Z, Z))
    Test.@test v ≈ out

    A = SymMps.SU2.singletmps(6)
    v1 = SymMps.e1!(A, Z)
    SymMps.SU2.e2!(A, kron(Z, Z))
    v2 = SymMps.e1!(A, Z)
    Test.@test v1 ≈ v2
end

Test.@testset "overlap" begin
    A = SymMps.SU2.singletmps(10)
    Test.@test SymMps.SU2.overlap(A, A) ≈ 1

    B = deepcopy(A)
    SymMps.apply!(A, SymMps.SU2.SWAP, 1)
    Test.@test SymMps.SU2.overlap(A, B) ≈ -1

    C = SymMps.sym_twosite_C(A, SymMps.SU2.SWAP)[1:2:end, 1:2:end]
    test = ones(Float64, 5, 5)
    Test.@test C ≈ test
end

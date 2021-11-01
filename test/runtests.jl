using Test
using Random
using LinearAlgebra
using SparseArrays

using Lanczos

@testset "lanczos_step! random matrices" begin

    @testset "dense" begin

        n = 100
        rng = MersenneTwister(42)
        H = randn(rng, n, n)
        H += H'


        m = 20 # max subspace dimension
        KS = KrylovSubspace(n, m)
        tol = 1e-10

        dt = 0.1
        ts = 0:dt:1
        state = randn(rng, n)
        state ./= norm(state)

        rl = zeros(ComplexF64, n, length(ts))
        rl[:,1] .= state
        red = zeros(ComplexF64, n, length(ts))
        red[:,1] .= state

        for i in 2:length(ts)
            ψ_i = @view rl[:,i-1]
            ψ_f = @view rl[:,i]
            lanczos_step!(ψ_f, KS, H, ψ_i, dt; tol=tol)
            t = ts[i]
            red[:,i] .= exp(-im*t*H)*state
        end
        @test all(rl .≈ red)

    end


    @testset "sparse" begin

        n = 100
        rng = MersenneTwister(42)
        H = sprandn(rng, n, n, 0.1)
        H += H'

        H_dense = Array(H)


        m = 20 # max subspace dimension
        KS = KrylovSubspace(n, m)
        tol = 1e-10

        dt = 0.1
        ts = 0:dt:1
        state = randn(rng, n)
        state ./= norm(state)

        rl = zeros(ComplexF64, n, length(ts))
        rl[:,1] .= state
        red = zeros(ComplexF64, n, length(ts))
        red[:,1] .= state

        for i in 2:length(ts)
            ψ_i = @view rl[:,i-1]
            ψ_f = @view rl[:,i]
            lanczos_step!(ψ_f, KS, H, ψ_i, dt; tol=tol)
            t = ts[i]
            red[:,i] .= exp(-im*t*H_dense)*state
        end
        @test all(rl .≈ red)

    end

end

module Lanczos

using LinearAlgebra

#export KrylovSubspace, KrylovSubspaceGPU
export KrylovSubspace
export lanczos_step, lanczos_step!


abstract type AbstractKrylovSubspace end

"""
Cache for Krylov subspace based methods.
"""
struct KrylovSubspace{rT, cT} <: AbstractKrylovSubspace
    m::Int
    V::Matrix{cT}
    α::Vector{rT}
    β::Vector{rT}
    #w::Vector{cT}
end

"""
KrylovSubspace(rT::Type, cT::Type, D, m)

Creates a cache for Krylov subspace based algorithms.
rT is the real type, while cT is the complex type of the caches.
D is the dimension of (Krylov) vectors, while m is the maximal
number of Krylov vectors that can be stored in the cache.

"""
KrylovSubspace(D::Int, m::Int) = KrylovSubspace(Float64, D, m)
function KrylovSubspace(rT::Type, D::Int, m::Int)
    cT = complex(rT)
    V = Matrix{cT}(undef, D, m)
    α = Vector{rT}(undef, m)
    β = Vector{rT}(undef, m-1)
    #w = Vector{cT}(undef, D)
    KrylovSubspace{rT, cT}(m, V, α, β)
end


#=
struct KrylovSubspaceGPU{rT, cT} <: AbstractKrylovSubspace
    m::Int
    V::CuMatrix{cT}
    α::Vector{rT}
    β::Vector{rT}
    #w::CuVector{cT}
end

KrylovSubspaceGPU(D::Int, m::Int) = KrylovSubspaceGPU(Float32, D, m)
function KrylovSubspaceGPU(rT::Type, D::Int, m::Int)
    cT = complex(rT)
    V = CuMatrix{cT}(undef, D, m)
    α = Vector{rT}(undef, m)
    β = Vector{rT}(undef, m-1)
    #w = CuVector{cT}(undef, D)
    KrylovSubspaceGPU{rT, cT}(m, V, α, β)
end
=#



"""
This functions are too specialized for general purpose usage.
"""
function lanczos_step(m::Int, H, ψ_i, dt::Real; vargs...)
    D = length(ψ_i)
    Ks = KrylovSubspace(D, m)
    ψ_f = Vector{ComplexF64}(undef, D)
    lanczos_step!(ψ_f, Ks, H, ψ_i, dt; vargs...)
end
"""
One iteration of the adaptive Lanczos algorithm overwriting the final state
ψ_f, a given Krylov cache structure Ks, the Hamiltonian H, an initial state
ψ_i and a time step dt.

The algorithms warns if the maximal Krylov dimension was reached.

Returns the final vector ψ_f and the used Krylov dimension.

ψ_i can be the same vector as ψ_f. ψ_i will then be overwritten

"""
function lanczos_step!(ψ_f::AbstractVector, Ks::AbstractKrylovSubspace,
    H, ψ_i::AbstractVector, dt::Real;
        krylov_dim_min=1, 
        tol=1e-14)

    krylov_dim_max = Ks.m
    V = Ks.V
    α = Ks.α
    β = Ks.β
    #w = Ks.w



    norm_ψ_i = norm(ψ_i)

    V[:,1] .= ψ_i ./ norm_ψ_i


    v1 = @view V[:,1]
    v2 = @view V[:,2]

    # Think of unrolling this matrix-vector mulitplication
    # into real and imaginary part, i.e. instead of allocating
    # complex vectors and keep real and imaginary part as seperate
    # vectors to hit BLAS gemv.
    # Especially important for LARGE dimensions
    # Update 06.04.20:
    #    when Julia is used with multiple threads this seems to be multithreaded
    #    already
    #mul!(w, H, v1)
    mul!(v2, H, v1)

    # we take the real part here because in exact arithmetics
    # dot(H * v1, v1) is real, as H is real symmetric
    # 09.08.2021
    # instead of real symmetric, hermitian should be enough here
    #
    # the dot function can be further optimized using
    # dot(x, A, y) syntax
    #
    # if this is not enough one can unroll the scalar product
    # and rewrite it in terms of real and imaginarty part
    # of v1
    # update 06.04.20:
    #   the code spends very little time in the dot product
    #α[1] = real(dot(w, v1))
    # update 01.11.2021:
    # the GPU code spents significant amount of time in the 
    # dot and norm functions
    α[1] = real(dot(v2, v1))

    #w .-= α[1] .* v1
    v2 .-= α[1] .* v1
    #β[1] = norm(w)
    β[1] = norm(v2)
    #V[:,2] .= w ./ β[1]
    v2 ./= β[1]

    # this is calculating krylov_dim_min Krylov vectors
    for j in 2:(krylov_dim_min-1)
        vj = @view V[:,j]
        vj_p1 = @view V[:,j+1]
        
        #mul!(w, H, vj)
        mul!(vj_p1, H, vj)

        # why real part see above
        #α[j] = real(dot(w, vj))
        α[j] = real(dot(vj_p1, vj))

        #vj_1 = @view V[:,j-1]
        vj_m1 = @view V[:,j-1]
        #w .-= α[j] .* vj .+ β[j-1] .* vj_1
        vj_p1 .-= α[j] .* vj .+ β[j-1] .* vj_m1
        #β[j] = norm(w)
        β[j] = norm(vj_p1)

        #V[:,j+1] .= w ./ β[j]
        vj_p1 ./= β[j]
    end

    # now do the adaptive part
    expidtT1 = nothing
    j = max(2, krylov_dim_min)
    while j < krylov_dim_max

        vj = @view V[:,j]
        vj_p1 = @view V[:,j+1]

        #mul!(w, H, vj)
        mul!(vj_p1, H, vj)

        # why real part see above
        #α[j] = real(dot(w, vj))
        α[j] = real(dot(vj_p1, vj))

        # check for convergence and abort if converged
        # we specialize the matrix structure to hit improved eigenvalue
        # decomposition (stegr)
        T = SymTridiagonal(α[1:j], β[1:j-1])
        F = eigen(T)
        expidtT1 = F.vectors * (exp.(-im*dt .* F.values) .* F.vectors[1,:])
        if abs(expidtT1[j]) < tol || j == krylov_dim_max
            break
        end

        #=
        vj_1 = @view V[:,j-1]
        w .-= α[j] .* vj .+ β[j-1] .* vj_1
        β[j] = norm(w)
        V[:,j+1] .= w ./ β[j]
        =#

        #vj_1 = @view V[:,j-1]
        vj_m1 = @view V[:,j-1]
        #w .-= α[j] .* vj .+ β[j-1] .* vj_1
        vj_p1 .-= α[j] .* vj .+ β[j-1] .* vj_m1
        #β[j] = norm(w)
        β[j] = norm(vj_p1)

        #V[:,j+1] .= w ./ β[j]
        vj_p1 ./= β[j]

        j += 1
    end

    V_krylov = if j >= krylov_dim_max
        @warn("max krylov dimesions reached, tol = $(abs(expidtT1[j-1]))")
        @view V[:, 1:(j-1)]
    else
        @view V[:, 1:j]
    end

    #V_krylov = @view V[:, 1:j]

    # create type instability
    # hope this is fine, because of function barrier mul!
    #expidtT1 = typeof(V_krylov) <: CuArray ? CuArray(expidtT1) : expidtT1
    mul!(ψ_f, V_krylov, expidtT1)


    #if !isapprox(norm_ψ_i, 1.;  atol=tol)
    ψ_f .*= norm_ψ_i
    #end

    ψ_f, j


end







end

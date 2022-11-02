# Lanczos.jl

This package implements a rudimentary version of the Lanczos algorithm in Julia. A matrix exponential $e^{tH}$ of a matrix $H$ is applied to a vector $x$ without calculating $e^{tH}$ explicitely. This is done by only evaluating $e^{tH}$ on a Krylov subspace. 

This package is for testing purposes only. If you are looking for a robust implementation of Krylov methods in Julia have a look at [KrylovKit.jl](https://github.com/Jutho/KrylovKit.jl).

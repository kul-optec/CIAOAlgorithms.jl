# # tests for: 1/N \sum_i (<x_i, Q_ix_i> + <x_i,q_i> + η dist(x_i,B_i) + g(\sum_i x_i) 

using Test
using LinearAlgebra
using CIAOAlgorithms
using ProximalOperators
using Base.Iterators: take

T = Float64

n = 2;
N = 3;
F = Vector{Sum}(undef, 0)
L = Vector{T}(undef, 0)
η = N * T(10)
box = IndBox(-T(2), T(2))
d = [T[1.0, 2.0], T[-1.0, 3.0], T[0.0, 10.0]]
for i = 1:N
    f2 = SqrDistL2(box, η) # soft box constraint
    Q = diagm(0 => d[i])
    f1 = Quadratic(Q, ones(T, n))
    push!(F, Sum(f1, f2))
    push!(L, opnorm(Q[i]) + η)
end
g = IndBox(-Inf, ones(T, n))   # sum_i x_i \leq 1 
x0 = zeros(T, n)

sum_star = [-5.136781609195401, -0.9333333333333327]


maxit = 1000
tol = 1e-4

@testset "Proshi" begin
    # sweeping 1, 2, 3 for randomined, cyclic and shuffled sampling strategies, respectively.

    ## test the solver
    # basic Proshi
    @testset "basic Proshi" for sweeping in collect(1:3)
        solver = CIAOAlgorithms.Proshi{T}(maxit = maxit, sweeping = sweeping)
        x_proshi, it_proshi = solver(x0, F = F, g = g, L = L, N = N)
        @test norm(sum(x_proshi) - sum_star, Inf) < tol
        @test eltype(x_proshi) == Array{T,1}
    end

    # basic Proshi with minibatch 
    vec_ref = [(1, 2), (2, 2), (3, 3)] # different samplings and batch sizes 
    @testset "Proshi_minibatch" for (sweeping, batch) in vec_ref
        solver = CIAOAlgorithms.Proshi{T}(
            maxit = maxit,
            sweeping = sweeping,
            minibatch = (true, batch),
        )
        x_proshi, it_proshi = solver(x0, F = F, g = g, L = L, N = N)
        @test norm(sum(x_proshi) - sum_star, Inf) < tol
        @test eltype(x_proshi) == Array{T,1}
    end

    # test with user defined stepsizes
    @testset "γ and L as scalars" begin
        @testset "randomized" begin
            γ = N / maximum(L)
            solver = CIAOAlgorithms.Proshi{T}(maxit = maxit, γ = γ)
            x_proshi, it_proshi = solver(x0, F = F, g = g, L = L, N = N)
            @test norm(sum(x_proshi) - sum_star, Inf) < tol
        end
        @testset "cyclic" begin
            solver = CIAOAlgorithms.Proshi{T}(maxit = maxit)
            x_proshi, it_proshi = solver(x0, F = F, g = g, L = maximum(L), N = N)
            @test norm(sum(x_proshi) - sum_star, Inf) < tol
        end
    end

    ## test the iterator 
    @testset "the iterator" for sweeping in collect(1:3)
        solver = CIAOAlgorithms.Proshi{T}(sweeping = sweeping)
        iter = CIAOAlgorithms.iterator(solver, x0, F = F, g = g, L = L, N = N)
        @test iter.x0 === x0

        for state in take(iter, 2)
            @test solution(state) === state.s
            @test eltype(solution(state)) == Array{T,1}
        end
    end
end

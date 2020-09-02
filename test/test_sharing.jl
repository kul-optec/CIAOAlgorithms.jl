# # tests for: 1/N \sum_i (<x_i, Q_ix_i> + <x_i,q_i> + η dist(x_i,B_i) + g(\sum_i x_i) 

# @testset "Lasso ($T)" for T in [Float32, Float64]
    using Test
    using LinearAlgebra
    using CIAOAlgorithms
    using ProximalOperators
    using Base.Iterators: take
    using Random

    Random.seed!(0)
    
    T = Float32
    R = real(T)

    n= 2;
    N= 3;
    F = Vector{Sum}(undef, 0)
    L = Vector{T}(undef, 0)
    η = N * R(10)
    box = IndBox(-R(2),R(2)) 
    for i in 1:N
        d_m = 2* rand(T, n) .- 1    
        f2 = SqrDistL2(box,η) # soft box constraint
        Q = diagm(0 => d_m)
        f1 = Quadratic(Q, ones(T, n))
        push!(F,Sum(f1,f2))
        push!(L,opnorm(Q[i])+η)
    end 
    g = IndBox(-Inf, ones(T, n))   # sum_i x_i \leq 1 
    x0 = zeros(T,n)

    sum_star = [-5.68848982666327, -5.3708267648106105]


    maxit = 2000
    tol = 1e-5

    @testset "Proshi" begin
        # sweeping 1, 2, 3 for randomined, cyclic and shuffled sampling strategies, respectively.

        ## test the solver
        # basic Proshi
        @testset "basic Proshi" for sweeping in collect(1:3)
            solver = CIAOAlgorithms.Proshi{R}(maxit = maxit, sweeping = sweeping)
            x_proshi, it_proshi = solver(x0, F = F, g = g,  L = L, N = N)
            @test norm(sum(x_proshi) - sum_star,Inf) < tol
        end

        # basic Proshi with minibatch 
        vec_ref = [(1, 2), (2, 2), (3, 3)] # different samplings and batch sizes 
        @testset "Proshi_minibatch" for (sweeping, batch) in vec_ref
            solver = CIAOAlgorithms.Proshi{R}(
                maxit = maxit,
                sweeping = sweeping,
                minibatch = (true, batch),
            )
            x_proshi, it_proshi = solver(x0, F = F, g = g, L = L, N = N)
            @test norm(sum(x_proshi) - sum_star,Inf) < tol
        end

        # test with user defined stepsizes
        @testset "γ and L as scalars" begin
            @testset "randomized" begin
                γ = N / maximum(L)
                solver = CIAOAlgorithms.Proshi{R}(maxit = maxit, γ = γ)
                x_proshi, it_proshi = solver(x0, F = F, g = g, L = L, N = N)
                @test norm(sum(x_proshi) - sum_star,Inf) < tol
            end
            @testset "cyclic" begin
                solver = CIAOAlgorithms.Proshi{R}(maxit = maxit)
                x_proshi, it_proshi = solver(x0, F = F, g = g, L = maximum(L), N = N)
                @test norm(sum(x_proshi) - sum_star,Inf) < tol
            end
        end

         ## test the iterator 
        @testset "the iterator" for sweeping in collect(1:3)

            solver = CIAOAlgorithms.Proshi{R}(sweeping = sweeping)
            iter = CIAOAlgorithms.iterator(solver, x0, F = F, g = g, L = L, N = N)
            @test iter.x0 === x0

            for state in take(iter, 2)
                @test solution(state) === state.s
                @test eltype(solution(state)) == Array{T,1}
            end
        end
    end
# end











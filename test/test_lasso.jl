using Test
using LinearAlgebra
using CIAOAlgorithms
using ProximalOperators
using ProximalAlgorithms
using Random


Random.seed!(0)

# #----------------------------------tests for lasso: 1/2\|Ax-b\|^2 + λ \|x\|_1-------------------------------------

@testset "Lasso" begin

    T = Float64
    I = Int64
    N, n = 4, 5 # A in R^{N x n}   
    p = 5 # nonzeros in the solution

    y_star = rand(N)
    y_star ./= norm(y_star) # y^star
    C = rand(N, n) .* 2 .- 1
    CTy = abs.(C' * y_star)
    # indices with decreasing order by abs
    perm = sortperm(CTy, rev = true) 

    rho, λ = 1.0, 1.0
    alpha = zeros(n)
    for i = 1:n
        if i <= p
            alpha[perm[i]] = λ / CTy[perm[i]]
        else
            alpha[perm[i]] = (CTy[perm[i]] < 0.1 * λ) ? λ : λ * rand() / CTy[perm[i]]
        end
    end
    A = C * diagm(0 => alpha)   # scaling the columns of Cin
    # generate the solution
    x_star = zeros(n)
    for i = 1:n
        if i <= p
            x_star[perm[i]] = rand() * rho / sqrt(p) * sign(dot(A[:, perm[i]], y_star))
        end
    end
    b = A * x_star + y_star

    # cost function
    cost_lasso(x) = norm(A * x - b) / 2 + λ * norm(x, 1)

    f_star = cost_lasso(x_star)

    # preparations for the solver 
    F = Vector{LeastSquares}(undef, 0)
    L = Vector{T}(undef, 0)
    for i = 1:N
        tempA = A[i:i, :]
        f = LeastSquares(tempA, b[i:i], Float64(N))
        Lf = opnorm(tempA)^2 * N
        push!(F, f)
        push!(L, Lf)
    end
    g = NormL1(λ)
    x0 = 10 * zeros(n)

    maxit = 1e5 |> Int64
    data_freq = 1
    tol = 1e-4

    # sweeping 1, 2, 3 for randomined, cyclic and shuffled sampling strategies, respectively.

    # basic finito
    @testset "nominal Finito" for sweeping in collect(1:3)
            solver = CIAOAlgorithms.Finito{T}(maxit = maxit, sweeping = sweeping)
            @time x_finito, it_finito = solver(F, g, x0, L = L, N = N)
            @test norm(cost_lasso(x_finito) - f_star) < tol
    end

    # limited memory finito 
    @testset "LFinito" for sweeping in collect(2:3)
        # @testset "cyclical" begin
            solver = CIAOAlgorithms.Finito{T}(maxit = maxit, sweeping = sweeping, LFinito = true)
            @time x_finito, it_finito = solver(F, g, x0, L = L, N = N)
            @test norm(cost_lasso(x_finito) - f_star) < tol
    end

    # adaptive variant 
    vec_ref = [(1, 3630), (2, 2772), (3, 2794)] #reference iteration numbers and sampling strategies
    @testset "adaptive finito" for (sweeping, it) in vec_ref
        # @testset "randomized" begin
            solver = CIAOAlgorithms.Finito{T}(maxit = maxit, tol = 1e-5, sweeping = sweeping, adaptive = true)
            @time x_finito, it_finito = solver(F, g, x0, L = L, N = N)
            @test it_finito < it
            @test norm(cost_lasso(x_finito) - f_star) < tol
    end
    
    # basic finito with minibatch 
    vec_ref = [(1, 2), (2, 2), (3, 3)] # different samplings and batch sizes 
    @testset "LFinito_minibatch" for (sweeping, batch) in vec_ref
            solver = CIAOAlgorithms.Finito{T}(
                maxit = maxit,
                sweeping = sweeping,
                minibatch = (true, batch),
            )
            @time x_finito, it_finito = solver(F, g, x0, L = L, N = N)
            @test norm(cost_lasso(x_finito) - f_star) < tol
    end

    # limited memory finito with minibatch 
    vec_ref = [(2, 1), (2, 2), (3, 3)] # different samplings and batch sizes 
    @testset "LFinito_minibatch" for (sweeping, batch) in vec_ref
            solver = CIAOAlgorithms.Finito{T}(
                maxit = maxit,
                sweeping = sweeping,
                LFinito = true,
                minibatch = (true, batch),
            )
            @time x_finito, it_finito = solver(F, g, x0, L = L, N = N)
            @test norm(cost_lasso(x_finito) - f_star) < tol
    end

    # test with user defined stepsizes
    @testset "γ and L as scalars" begin
        @testset "randomized" begin
            γ = N / maximum(L)
            solver = CIAOAlgorithms.Finito{T}(maxit = maxit, γ = γ)
            @time x_finito, it_finito = solver(F, g, x0, L = L, N = N)
            @test norm(cost_lasso(x_finito) - f_star) < tol
        end
        @testset "cyclic" begin
            solver = CIAOAlgorithms.Finito{T}(maxit = maxit)
            @time x_finito, it_finito = solver(F, g, x0, L = maximum(L), N = N)
            @test norm(cost_lasso(x_finito) - f_star) < tol
        end
    end


end

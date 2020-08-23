# # tests for lasso: 1/2\|Ax-b\|^2 + λ \|x\|_1

@testset "Lasso ($T)" for T in [Float32, Float64]
    using Test
    using LinearAlgebra
    using CIAOAlgorithms
    using ProximalOperators
    using Random

    Random.seed!(0)

    R = real(T)

    N, n = 6, 5 # A in R^{N x n}   
    p = 2 # nonzeros in the solution

    y_star = rand(T, N)
    y_star ./= norm(y_star) # y^star
    C = rand(T, N, n) .* 2 .- 1
    CTy = abs.(C' * y_star)
    # indices with decreasing order by abs
    perm = sortperm(CTy, rev = true)

    rho, λ = R(10), R(1)
    alpha = zeros(T, n)
    for i = 1:n
        if i <= p
            alpha[perm[i]] = λ / CTy[perm[i]]
        else
            alpha[perm[i]] = (CTy[perm[i]] < 0.1 * λ) ? λ : λ * rand(T) / CTy[perm[i]]
        end
    end
    A = C * diagm(0 => alpha)   # scaling the columns of Cin
    # generate the solution
    x_star = zeros(T, n)
    for i = 1:n
        if i <= p
            x_star[perm[i]] = rand(T) * rho / sqrt(p) * sign(dot(A[:, perm[i]], y_star))
        end
    end
    b = A * x_star + y_star

    # cost function
    cost_lasso(x) = norm(A * x - b) / 2 + λ * norm(x, 1)

    f_star = cost_lasso(x_star)

    # preparations for the solver 
    F = Vector{LeastSquares}(undef, 0)
    L = Vector{R}(undef, 0)
    for i = 1:N
        tempA = A[i:i, :]
        f = LeastSquares(tempA, b[i:i], R(N))
        Lf = opnorm(tempA)^2 * N
        push!(F, f)
        push!(L, Lf)
    end
    g = NormL1(λ)
    x0 = zeros(T, n)

    maxit = 5000
    tol = 1e-3

    @testset "Finito" begin
        # sweeping 1, 2, 3 for randomined, cyclic and shuffled sampling strategies, respectively.

        ## test the solver
        @testset "with solver" begin
            # basic finito
            @testset "basic Finito" for sweeping in collect(1:3)
                solver = CIAOAlgorithms.Finito{R}(maxit = maxit, sweeping = sweeping)
                x_finito, it_finito = solver(F, g, x0, L = L, N = N)
                @test norm(cost_lasso(x_finito) - f_star) < tol
            end

            # limited memory finito 
            @testset "LFinito" for sweeping in collect(2:3)
                # @testset "cyclical" begin
                solver = CIAOAlgorithms.Finito{R}(
                    maxit = maxit,
                    sweeping = sweeping,
                    LFinito = true,
                )
                x_finito, it_finito = solver(F, g, x0, L = L, N = N)
                @test norm(cost_lasso(x_finito) - f_star) < tol
            end

            # adaptive variant 
            @testset "adaptive finito" for sweeping in collect(1:3)
                solver = CIAOAlgorithms.Finito{R}(
                    maxit = maxit,
                    tol = R(1e-5),
                    sweeping = sweeping,
                    adaptive = true,
                )
                x_finito, it_finito = solver(F, g, x0, L = L, N = N)
                # @test it_finito < it
                @test norm(cost_lasso(x_finito) - f_star) < tol
            end

            # basic finito with minibatch 
            vec_ref = [(1, 2), (2, 2), (3, 3)] # different samplings and batch sizes 
            @testset "Finito_minibatch" for (sweeping, batch) in vec_ref
                solver = CIAOAlgorithms.Finito{R}(
                    maxit = maxit,
                    sweeping = sweeping,
                    minibatch = (true, batch),
                )
                x_finito, it_finito = solver(F, g, x0, L = L, N = N)
                @test norm(cost_lasso(x_finito) - f_star) < tol
            end

            # limited memory finito with minibatch 
            vec_ref = [(2, 1), (2, 2), (3, 3)] # different samplings and batch sizes 
            @testset "LFinito_minibatch" for (sweeping, batch) in vec_ref
                solver = CIAOAlgorithms.Finito{R}(
                    maxit = maxit,
                    sweeping = sweeping,
                    LFinito = true,
                    minibatch = (true, batch),
                )
                x_finito, it_finito = solver(F, g, x0, L = L, N = N)
                @test norm(cost_lasso(x_finito) - f_star) < tol
            end

            # test with user defined stepsizes
            @testset "γ and L as scalars" begin
                @testset "randomized" begin
                    γ = N / maximum(L)
                    solver = CIAOAlgorithms.Finito{R}(maxit = maxit, γ = γ)
                    x_finito, it_finito = solver(F, g, x0, L = L, N = N)
                    @test norm(cost_lasso(x_finito) - f_star) < tol
                end
                @testset "cyclic" begin
                    solver = CIAOAlgorithms.Finito{R}(maxit = maxit)
                    x_finito, it_finito = solver(F, g, x0, L = maximum(L), N = N)
                    @test norm(cost_lasso(x_finito) - f_star) < tol
                end
            end
        end

        ## test the one iteration update function
        @testset "one iterate" begin

            solver = CIAOAlgorithms.Finito{R}(sweeping = 1)
            problem = CIAOAlgorithms.Finito_problem(solver, F, g, x0, L = L, N = N)
            @test problem.sol === x0
            @test problem.cnt === 0

            for k = 1:2
                CIAOAlgorithms.update!(problem)
                @test problem.sol === problem.state.z
                @test k === problem.cnt
            end
            @testset "basic Finito" for sweeping in collect(1:3)
                solver = CIAOAlgorithms.Finito{R}(sweeping = sweeping)
                problem = CIAOAlgorithms.Finito_problem(solver, F, g, x0, L = L, N = N)
                for k = 1:maxit
                    CIAOAlgorithms.update!(problem)
                end
                @test norm(cost_lasso(problem.sol) - f_star) < tol
            end

            # limited memory finito 
            @testset "LFinito" for sweeping in collect(2:3)
                # @testset "cyclical" begin
                solver = CIAOAlgorithms.Finito{R}(sweeping = sweeping, LFinito = true)
                problem = CIAOAlgorithms.Finito_problem(solver, F, g, x0, L = L, N = N)
                for k = 1:maxit
                    CIAOAlgorithms.update!(problem)
                end
                @test norm(cost_lasso(problem.sol) - f_star) < tol
            end

            # adaptive variant 
            @testset "adaptive finito" for sweeping in collect(1:3)
                solver = CIAOAlgorithms.Finito{R}(
                    tol = R(1e-5),
                    sweeping = sweeping,
                    adaptive = true,
                )
                problem = CIAOAlgorithms.Finito_problem(solver, F, g, x0, L = L, N = N)
                for k = 1:maxit
                    CIAOAlgorithms.update!(problem)
                end
                @test norm(cost_lasso(problem.sol) - f_star) < tol
            end
        end


    end


    @testset "SVRG" begin

        ## test the solver
        @testset "with solver" begin
            γ = 1 / (7 * maximum(L))
            @testset "SVRG-Base" begin
                solver = CIAOAlgorithms.SVRG{T}(maxit = maxit, γ = γ)
                x_SVRG, it_SVRG = solver(F, g, x0, N = N)
                @test norm(cost_lasso(x_SVRG) - f_star) < tol
            end
            @testset "SVRG++" begin
                solver = CIAOAlgorithms.SVRG{T}(maxit = 16, γ = γ, m = 1, plus = true)
                x_SVRG, it_SVRG = solver(F, g, x0, N = N)
                @test norm(cost_lasso(x_SVRG) - f_star) < tol
            end
        end

        ## test the one iteration update function
        @testset "one iterate" begin
            γ = 1 / (7 * maximum(L))
            @testset "SVRG-Base" begin
                solver = CIAOAlgorithms.SVRG{T}(γ = γ)
                problem = CIAOAlgorithms.SVRG_problem(solver, F, g, x0, N = N)
                for k = 1:maxit
                    CIAOAlgorithms.update!(problem)
                end
                @test norm(cost_lasso(problem.sol) - f_star) < tol
            end

            @testset "SVRG++" begin
                solver = CIAOAlgorithms.SVRG{T}(γ = γ, m = 1, plus = true)
                problem = CIAOAlgorithms.SVRG_problem(solver, F, g, x0, N = N)
                for k = 1:16
                    CIAOAlgorithms.update!(problem)
                end
                @test norm(cost_lasso(problem.sol) - f_star) < tol
            end

        end
    end
end

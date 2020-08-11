# # tests for logistic regression: sum_i log(1+exp(-b_i<a_i,x>)) + λ \|x\|_1

@testset "logistic-l1" begin

    using LinearAlgebra
    using CIAOAlgorithms
    using ProximalOperators


    T = Float64
    # create the two classes 
    x_class1 = [
        5.1 3.5 1.4 0.2 1.0
        4.9 3.0 1.4 0.2 1.0
        4.7 3.2 1.3 0.2 1.0
        4.6 3.1 1.5 0.2 1.0
    ]
    x_class2 = [
        5.7 3.0 4.2 1.2 1.0
        5.7 2.9 4.2 1.3 1.0
        6.2 2.9 4.3 1.3 1.0
        5.1 2.5 3.0 1.1 1.0
    ]

    xs = [x_class1; x_class2]   # taking 500 sample from each
    ys = [fill(1.0, size(x_class1, 1)); fill(-1.0, size(x_class2, 1))]

    # preparations for the solver 
    x_star = [0.0, 2.5591213030017443, -3.062235206001081, 0.0, 0.0]

    N, n = size(ys, 1), size(xs, 2)

    F = Vector{Precompose}(undef, 0)
    L = Vector{T}(undef, 0)
    for i = 1:N
        f = Precompose(LogisticLoss([ys[i]], 1.0), reshape(xs[i, :], 1, n), 1.0)
        push!(F, f)
        # compute L
        Lf = 0.25 * norm(xs[i, :])^2
        push!(L, Lf)
    end

    # compute L for the nonadaptive case 
    lam = 0.1 / N
    g = NormL1(lam)
    x0 = 10 * zeros(n)

    maxit = 1e5 |> Int64
    tol = 1e-3

    @testset "Finito" begin

        # sweeping 1, 2, 3 for randomined, cyclic and shuffled sampling strategies, respectively.

        # basic finito  
        @testset "nominal Finito" for sweeping in collect(1:3)
            solver = CIAOAlgorithms.Finito{T}(maxit = maxit, sweeping = sweeping)
            x_finito, it_finito = solver(F, g, x0, L = L, N = N)
            @test norm(x_finito - x_star) < tol
        end

        # limited memory finito 
        @testset "LFinito" for sweeping in collect(2:3)
            # @testset "cyclical" begin
            solver =
                CIAOAlgorithms.Finito{T}(maxit = maxit, sweeping = sweeping, LFinito = true)
            x_finito, it_finito = solver(F, g, x0, L = L, N = N)
            @test norm(x_finito - x_star) < tol
        end

        # basic finito with minibatch 
        vec_ref = [(1, 2), (2, 2), (3, 3)] # different samplings and batch sizes 
        @testset "LFinito_minibatch" for (sweeping, batch) in vec_ref
            solver = CIAOAlgorithms.Finito{T}(
                maxit = maxit,
                sweeping = sweeping,
                minibatch = (true, batch),
            )
            x_finito, it_finito = solver(F, g, x0, L = L, N = N)
            @test norm(x_finito - x_star) < tol
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
            x_finito, it_finito = solver(F, g, x0, L = L, N = N)
            @test norm(x_finito - x_star) < tol
        end

        # test with user defined stepsizes
        @testset "γ and L as scalars" begin
            @testset "randomized" begin
                γ = N / maximum(L)
                solver = CIAOAlgorithms.Finito{T}(maxit = maxit, γ = γ)
                x_finito, it_finito = solver(F, g, x0, L = L, N = N)
                @test norm(x_finito - x_star) < tol
            end
            @testset "cyclic" begin
                solver = CIAOAlgorithms.Finito{T}(maxit = maxit)
                x_finito, it_finito = solver(F, g, x0, L = maximum(L), N = N)
                @test norm(x_finito - x_star) < tol
            end
        end
    end


    @testset "SVRG" begin
        γ = 1 / (10 * maximum(L))
        @testset "SVRG-Base" begin
            solver = CIAOAlgorithms.SVRG{T}(maxit = maxit, γ = γ)
            x_SVRG, it_SVRG = solver(F, g, x0, N = N)
            @test norm(x_SVRG - x_star) < tol
        end
        @testset "SVRG++" begin
            solver = CIAOAlgorithms.SVRG{T}(maxit = 16, γ = γ, m = N, plus = true)
            x_SVRG, it_SVRG = solver(F, g, x0, N = N)
            @test norm(x_SVRG - x_star) < tol
        end
    end

end

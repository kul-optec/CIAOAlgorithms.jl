using Test
using LinearAlgebra
using StoInc
using ProximalOperators
using RecursiveArrayTools: ArrayPartition, unpack
using ProximalAlgorithms
using Random


#TODO: 
#### 

Random.seed!(0) 

# # for debugging the code----------------------------------Do not modify below-------------------------------------
@testset "test using quadratic" begin
	T = Float64
	A = diagm(0 => 1.0:4.0)	
	b = T[1.0, 2.0, 3.0, 4.0]
	m, n = size(A)
	f = LeastSquares(A, b)	
	Lf = opnorm(A)^2	
	g = IndFree()
	N =100
	F = fill(f,(N,)) # feed the same function for testing 
	L = fill(Lf, (N,))
	mf = eigmin(A'*A)  
	μ = fill(mf, (N,))
	tol=1e-5 
	maxit = 1e5 |> Int
	@testset "finito" begin
		@testset "randomized_minibach_one" begin 
				solver = CIAOAlgorithms.Finito{T}(maxit=20, minibatch =(true,1), sweeping =2 )    		
				x0 = 10*randn(n)
				@time x_finito_mb, it_finito_mb = solver(F, g, x0, L=L, N=N)

				solver = CIAOAlgorithms.Finito{T}(maxit=20,sweeping =2 )    		
				@time x_finito, it_finito = solver(F, g, x0, L=L, N=N)
				@test norm(x_finito - x_finito_mb) < 1e-8
		end 
		@testset "LFinito_minibach_one" begin 
				solver = CIAOAlgorithms.Finito{T}(maxit=200, minibatch =(true,1), sweeping =2, LFinito = true)    		
				x0 = 10*randn(n)
				@time x_finito_mb, it_finito_mb = solver(F, g, x0, L=L, N=N)

				solver = CIAOAlgorithms.Finito{T}(maxit=200, sweeping =2, LFinito = true)  		
				@time x_finito, it_finito = solver(F, g, x0, L=L, N=N)
				@test norm(x_finito - x_finito_mb) < 1e-8
		end 
		@testset "randomized" begin 
				solver = CIAOAlgorithms.Finito{T}(maxit=maxit, minibatch =(true,N) )    		
				x0 = 10*randn(n)
				@time x_finito, it_finito = solver(F, g, x0, L=L, N=N)
				@test norm(x_finito - ones(n)) < tol
		end 
		@testset "cyclical" begin 		
				solver = CIAOAlgorithms.Finito{T}(maxit=maxit, sweeping=2,  minibatch =(true,2) )    		
				x0 = 10*randn(n)  
				@time x_finito, it_finito = solver(F, g, x0, L=L, N=N)
				@test norm(x_finito - ones(n)) < tol
		end	
		@testset "shuffled" begin 
				solver = CIAOAlgorithms.Finito{T}(maxit=maxit, sweeping=3,  minibatch =(true,3) )    				
				x0 = 10*randn(n)
				@time x_finito, it_finito = solver(F, g, x0, L=L, N=N)
				@test norm(x_finito - ones(n)) < tol
		end	
	end 	
	@testset "γ-provided by the user" begin
		@testset "randomized-stepsize_provided" begin 
				γ = fill(N/Lf, (N,))
				solver = CIAOAlgorithms.Finito{T}(maxit=maxit, γ = γ,  minibatch =(true,2), sweeping =1 )    		
				x0 = 10*randn(n)
				@time x_finito, it_finito = solver(F, g, x0, L=L, N=N)
				@test norm(x_finito - ones(n)) < tol
		end 
		@testset "randomized-stepsize_provided" begin 
				γ = N/Lf
				solver = CIAOAlgorithms.Finito{T}(maxit=maxit, γ = γ,  minibatch =(true,2) )    		
				x0 = 10*randn(n)
				@time x_finito, it_finito = solver(F, g, x0, L=L, N=N)
				@test norm(x_finito - ones(n)) < tol
		end 
	end
end



@testset "L2-LogReg" begin
	T = Float64
	A = T[
		1.0  -2.0   3.0  -4.0  5.0;
		2.0  -1.0   0.0  -1.0  3.0;
		-1.0   0.0   4.0  -3.0  2.0;
		-1.0  -1.0  -1.0   1.0  3.0
	]
	b = T[1.0, 2.0, 3.0, 4.0]
	N, n = size(A)
	lam = 0.1
	g = SqrNormL2(lam)
    x_star = T[-0.45169112670858613,  -0.5906273249880297, 0.33408279533377466, -0.24527747237507025, 2.177200385771634]
	F = Vector{Precompose}(undef, 0)
	L = Vector{T}(undef, 0)
	for i in 1:N
		f = Precompose(LogisticLoss(ones(1), Float64(N)), reshape(A[i,:],1,n), 1.0, -reshape([b[i]],1))
		Lf = N* norm(A[i,:])^2/4
		push!(F,f)
		push!(L,Lf)
	end  
	maxit = 1e5 |>Int
		@testset "L2-LogReg (1 by n)" begin
			x0 = 1*randn(n)
			d =N # minibatch size
			solver = CIAOAlgorithms.Finito{T}(maxit=maxit, tol=1e-6, verbose=true, freq=10000)
			@time x_finito, it_finito = solver(F, g, x0, N=N, L=L)
			@test norm(x_finito-x_star) <= 1e-4 
		end 
		@testset "L2-LogReg mini batch 2" begin
			x0 = 1*randn(n)
			d = 4
			solver = CIAOAlgorithms.Finito{T}(maxit=maxit, tol=1e-6, verbose=true, freq=10000, minibatch= (true , d))
			@time x_finito, it_finito = solver(F, g, x0, N=N, L=L)
			@test norm(x_finito-x_star) <= 1e-4 
		end 
		@testset "L2-LogReg mini batch 2" begin
			x0 = 1*randn(n)
			d = 4
			solver = CIAOAlgorithms.Finito{T}(maxit=maxit, tol=1e-6, verbose=true, freq=10000, LFinito=true,
			 minibatch= (true , d))
			@time x_finito, it_finito = solver(F, g, x0, N=N, L=L)
			@test norm(x_finito-x_star) <= 1e-4 
		end 
	end

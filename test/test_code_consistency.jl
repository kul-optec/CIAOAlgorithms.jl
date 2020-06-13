using Test
using LinearAlgebra
using CIAOAlgorithms
using ProximalOperators
using RecursiveArrayTools: ArrayPartition, unpack
using ProximalAlgorithms
using Random


#TODO: 
#### add lasso, logl2, logl1

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
	N =10
	F = fill(f,(N,)) # feed the same function for testing 
	L = fill(Lf, (N,))
	mf = eigmin(A'*A)  
	μ = fill(mf, (N,))
	tol=1e-5 
	maxit = 15000
	@testset "nominal finito" begin
		@testset "nominal-randomized" begin 
				solver = CIAOAlgorithms.Finito{T}(maxit=maxit)    		
				x0 = 10*randn(n)
				@time x_finito, it_finito = solver(F, g, x0, L=L, N=N)
				@test norm(x_finito - ones(n)) < tol
		end 
		@testset "nominal-cyclical" begin 		
				solver = CIAOAlgorithms.Finito{T}(maxit=maxit, sweeping=2)    		
				x0 = 10*randn(n)  
				@time x_finito, it_finito = solver(F, g, x0, L=L, N=N)
				@test norm(x_finito - ones(n)) < tol
		end	
		@testset "nominal-shuffled" begin 
				solver = CIAOAlgorithms.Finito{T}(maxit=maxit, sweeping=3)    				
				x0 = 10*randn(n)
				@time x_finito, it_finito = solver(F, g, x0, L=L, N=N)
				@test norm(x_finito - ones(n)) < tol
		end	
	end 	
	@testset "extended nominal" begin	
		@testset "nominal-extended-randomized" begin 
				solver = CIAOAlgorithms.Finito{T}(maxit=maxit, tol=1e-5,  extended = true)    		
				x0 = 10*randn(n)
				@time x_finito, it_finito = solver(F, g, x0, L=L, N=N)	 
				@test it_finito == 1703
		end 
		@testset "nominal-extended-cyclical" begin 		
				solver = CIAOAlgorithms.Finito{T}(maxit=maxit, tol=1e-5, sweeping=2,  extended = true)    		
				x0 = 10*randn(n)  
				@time x_finito, it_finito = solver(F, g, x0, L=L, N=N)
				@test it_finito ==980  
		end	
		@testset "nominal-extended-shuffled" begin 
				solver = CIAOAlgorithms.Finito{T}(maxit=maxit, tol=1e-5, sweeping=3, extended = true)    				
				x0 = 10*randn(n)
				@time x_finito, it_finito = solver(F, g, x0, L=L, N=N)
				@test it_finito ==1107   
		end	
		@testset "nominal-extended-weighted-probabilities" begin 	
				solver = CIAOAlgorithms.Finito{T}(maxit=maxit, tol=1e-5, sweeping=4, extended = true)    			
				x0 = 10*randn(n)
				@time x_finito, it_finito = solver(F, g, x0, L=L, N=N, μ =μ)
				@test it_finito ==3201   
		end	
	end
	@testset "LFinito" begin
		@testset "nominal-extended-cyclical" begin 		
				solver = CIAOAlgorithms.Finito{T}(maxit=maxit, sweeping=2,  LFinito = true)    		
				x0 = 10*randn(n)  
				@time x_finito, it_finito = solver(F, g, x0, L=L, N=N)
				@test norm(x_finito - ones(n)) < tol  
		end	
		@testset "nominal-extended-shuffled" begin 
				solver = CIAOAlgorithms.Finito{T}(maxit=maxit, sweeping=3, LFinito = true)    				
				x0 = 10*randn(n)
				@time x_finito, it_finito = solver(F, g, x0, L=L, N=N)
				@test norm(x_finito - ones(n)) < tol   
		end	
	end
	@testset "adaptive finito" begin
		@testset "randomized" begin 
				solver = CIAOAlgorithms.Finito{T}(maxit=maxit, tol=1e-5, adaptive = true) 
				x0 = 10*randn(n)
				@time x_finito, it_finito = solver(F, g, x0, L=L, N=N)   
				@test it_finito == 1095
		end 
		@testset "cyclic" begin 	
				solver = CIAOAlgorithms.Finito{T}(maxit=maxit, tol=1e-5, adaptive = true, sweeping=2) 	
				x0 = 10*randn(n)
				@time x_finito, it_finito = solver(F, g, x0, L=L, N=N)
				 @test it_finito == 625  
		end	
		@testset "shuffled" begin 		
				solver = CIAOAlgorithms.Finito{T}(maxit=maxit, tol=1e-5, adaptive = true, sweeping=3) 	
				x0 = 10*randn(n)
				@time x_finito, it_finito = solver(F, g, x0, L=L, N=N)
				 @test it_finito == 708   
		end	
	end 
	@testset "γ-provided by the user" begin
		@testset "nominal-randomized-stepsize_provided" begin 
				γ = fill(N/Lf, (N,))
				solver = CIAOAlgorithms.Finito{T}(maxit=maxit, γ = γ)    		
				x0 = 10*randn(n)
				@time x_finito, it_finito = solver(F, g, x0, L=L, N=N)
				@test norm(x_finito - ones(n)) < tol
		end 
		@testset "nominal-randomized-stepsize_provided" begin 
				γ = N/Lf
				solver = CIAOAlgorithms.Finito{T}(maxit=maxit, γ = γ)    		
				x0 = 10*randn(n)
				@time x_finito, it_finito = solver(F, g, x0, L=L, N=N)
				@test norm(x_finito - ones(n)) < tol
		end 
	end
	@testset "LFinito_minibatch" begin
		@testset "nominal-cyclical" begin 		
				solver = CIAOAlgorithms.Finito{T}(maxit=maxit, sweeping=2,  LFinito = true, minibatch = (true, 2))    		
				x0 = 10*randn(n)  
				@time x_finito, it_finito = solver(F, g, x0, L=L, N=N)
				@test norm(x_finito - ones(n)) < tol  
		end	
		@testset "nominal -shuffled" begin 
				solver = CIAOAlgorithms.Finito{T}(maxit=maxit, sweeping=3, LFinito = true, minibatch = (true, 5))    				
				x0 = 10*randn(n)
				@time x_finito, it_finito = solver(F, g, x0, L=L, N=N)
				@test norm(x_finito - ones(n)) < tol   
		end	
	end
end





# generating an example with different condition numbers 
@testset "test sum of quadratic and l1_extended" begin
		T = Float64
		I = Int64
		n= 5;
		N= 20 
		xi = 1   #greater than 0 

		F= Vector{LeastSquares}(undef, 0)
		L = Vector{T}(undef, 0)
		μ = Vector{T}(undef, 0)
		for i in 1:N
			a_M = 1.0 .+ (10^xi-1) * rand(floor(n/2) |> I)
			a_m = 0.1^xi .+ (1-0.1^xi) * rand(n-floor(n/2)|> I)
			A = diagm(0 => [a_m; a_M])
			b = 1e2*rand(n)
			m, n = size(A)
			f = LeastSquares(A, b)	
			Lf = opnorm(A)^2	
			mf = eigmin(A'*A) 	
			push!(F,f)
			push!(L,Lf)
			push!(μ,mf)
		end
		lam = 0.1
		g = NormL1(lam)
		maxit = 1e6 |>Int
	@testset "nominal-extended-randomized" begin 
			solver = CIAOAlgorithms.Finito{T}(maxit=maxit |> I, tol=1e-5,freq=1e5 |> I, sweeping=1, extended = true)    				
			x0 = 10*randn(n)
			@time x_finito, it_finito = solver(F, g, x0, L=L, N=N)   
			@test it_finito == 59863
	end 
	@testset "nominal-extended-stepsizes" begin 
			solver = CIAOAlgorithms.Finito{T}(maxit=maxit |> I, tol=1e-5, freq=1e5 |> I, single_stepsize = true, extended = true)    						
			x0 = 10*randn(n)
			@time x_finito, it_finito = solver(F, g, x0, L=L, N=N)
			@test it_finito == 89967
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
	m, n = size(A)
	lam = 0.1
	g = SqrNormL2(lam)
    x_star = T[-0.45169112670858613,  -0.5906273249880297, 0.33408279533377466, -0.24527747237507025, 2.177200385771634]	
	maxit = 1e6 |> Int
	@testset "L2-LogReg (1 by n)" begin
		x0 = 1*randn(n)
		N =m
		F = Vector{Precompose}(undef, 0)
		for i in 1:N
			f = Precompose(LogisticLoss(ones(1), Float64(N)), reshape(A[i,:],1,n), 1.0, -reshape([b[i]],1))
			push!(F,f)
		end  
		solver = CIAOAlgorithms.Finito{T}(maxit=maxit, tol=1e-6, verbose=true, freq=100, adaptive=true)
		@time x_finito, it_finito = solver(F, g, x0, N=N)
		@test norm(x_finito-x_star) <= 1e-4 
		@test it_finito == 2021
	end 
	@testset "L2-LogReg (1 by n) no g" begin
		x0 = 1*randn(n)
		N =m
		F = Vector{Sum}(undef, 0)
		for i in 1:N
			f = Precompose(LogisticLoss(ones(1), Float64(N)), reshape(A[i,:],1,n), 1.0, -reshape([b[i]],1))
			push!(F,Sum(f,SqrNormL2(lam)))
		end  
		g = IndFree()
		solver = CIAOAlgorithms.Finito{T}(maxit=maxit, tol=1e-6, verbose=true, freq=1000, adaptive=true)
		@time x_finito, it_finito = solver(F, g, x0, N=N)
		@test norm(x_finito-x_star) <= 1e-4 
	end 
	@testset "L2-LogReg (2 by n)" begin
		solver = CIAOAlgorithms.Finito{T}(maxit=maxit, tol=1e-6, verbose=true, freq=100, adaptive=true)
		x0 = 1*randn(n)
		N = 2; 
		g = SqrNormL2(lam)
		F = [Precompose(LogisticLoss(ones(2), 2.0), A[3:4,:], 1.0, -b[3:4]), Precompose(LogisticLoss(ones(2), 2.0), A[1:2,:], 1.0, -b[1:2])]
		@time x_finito, it_finito = solver(F, g, x0, N=N)
		@test norm(x_finito-x_star) <= 1e-4 
		@test it_finito == 857
	end

	@testset "L2-LogReg (4 by n)" begin
		solver = CIAOAlgorithms.Finito{T}(maxit=maxit, tol=1e-6, verbose=true, freq=100, adaptive=true)
		x0 = 1*randn(n)
		N = 1; 
		f = Precompose(LogisticLoss(ones(m), 1.0), A, 1.0, -b)
		F = fill(f,(N,)) # feed the same function for testing 
		@time x_finito, it_finito = solver(F, g, x0, N=N)
		@test norm(x_finito-x_star) <= 1e-4 
		@test it_finito == 824
	end

	@testset "L2-LogReg (4 by n)" begin
		solver = CIAOAlgorithms.Finito{T}(  sweeping =2,
			 maxit=maxit, tol=1e-6, verbose=true, freq=100, adaptive=true)
		x0 = 1*randn(n)
		N = 1; 
		f = Precompose(LogisticLoss(ones(m), 1.0), A, 1.0, -b)
		F = fill(f,(N,)) # feed the same function for testing 
		@time x_finito, it_finito = solver(F, g, x0, N=N)
		@test norm(x_finito-x_star) <= 1e-4 
		@test it_finito == 824
	end
	 @testset "L2-LogReg (1 by n)-cyclical-adaptive" begin
	 	solver = CIAOAlgorithms.Finito{T}(  sweeping=2,
			 maxit=100000, tol=1e-6, verbose=true, freq=10000, adaptive=true)
		x0 = 10*randn(n)
		N =m
		F = Vector{Precompose}(undef, 0)
		for i in 1:N
			f = Precompose(LogisticLoss(ones(1), Float64(N)), reshape(A[i,:],1,n), 1.0, -reshape([b[i]],1))
			push!(F,f)
		end  
		@time x_finito, it_finito = solver(F, g, x0, N=N)
		@test norm(x_finito-x_star) <= 1e-4 
		@test it_finito == 908
	end
	@testset "L2-LogReg (1 by n)-shuffled-adaptive" begin
		solver = CIAOAlgorithms.Finito{T}(sweeping=3,
			 maxit=100000, tol=1e-6, verbose=true, freq=10000, adaptive=true)
		x0 = 10*randn(n)
		N =m
		F = Vector{Precompose}(undef, 0)
		for i in 1:N
			f = Precompose(LogisticLoss(ones(1), Float64(N)), reshape(A[i,:],1,n), 1.0, -reshape([b[i]],1))
			push!(F,f)
		end  
		@time x_finito, it_finito = solver(F, g, x0, N=N)
		@test norm(x_finito-x_star) <= 1e-4 
		@test it_finito == 935
	end
end



@testset "test using quadratic" begin
				T = Float64
				I = Int64
				n= 5;
				N= 10
				xi = 1   #greater than 0 

				F= Vector{LeastSquares}(undef, 0)
				L = Vector{T}(undef, 0)
				μ = Vector{T}(undef, 0)
				for i in 1:N
					a_M = 1.0 .+ (10^xi-1) * rand(floor(n/2) |> I)
					a_m = 0.1^xi .+ (1-0.1^xi) * rand(n-floor(n/2)|> I)
					A = diagm(0 => [a_m; a_M])
					b = 1e2*rand(n)
					# m, n = size(A)
					f = LeastSquares(A, b)	
					Lf = opnorm(A)^2	
					mf = eigmin(A'*A) 	
					push!(F,f)
					push!(L,Lf)
					push!(μ,mf)
				end
 				lam = 0.1
				g = NormL1(lam)
				maxit = 1e6 |> Int
		@testset "Test randomized" begin
				solver = CIAOAlgorithms.Finito{Float64}( maxit=1e7 |> I, tol=1e-5, verbose=true, freq=1e6 |> I, 
				  extended = true, sweeping =1) 		
				x0 = 10*randn(n)
				@time x_finito, it_finito = solver(F, g, x0, L=L, N=N, μ = μ)   
				@test it_finito == 35559
		end 
		@testset "weighted-probabilities" begin 
				solver = CIAOAlgorithms.Finito{Float64}(maxit=1e7 |> I, tol=1e-5, verbose=true, freq=1e6 |> I, 
				  extended= true, sweeping =4) 	
				x0 = 10*randn(n)
				@time x_finito, it_finito = solver(F, g, x0, L=L, N=N, μ = μ) 
				@test it_finito == 131991
		end 
end

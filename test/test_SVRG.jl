using Test
using LinearAlgebra
using StoInc
using ProximalOperators
# using RecursiveArrayTools: ArrayPartition, unpack
using ProximalAlgorithms
using Random


#TODO: 
#### fix nominal ones, currently in log l2 all is adaptive 



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
	maxit = 3000 |> Int
		x0 = randn(n)
		N =m
		F = Vector{Precompose}(undef, 0)
		for i in 1:N
			f = Precompose(LogisticLoss(ones(1), Float64(N)), reshape(A[i,:],1,n), 1.0, -reshape([b[i]],1))
			push!(F,f)
		end  
		L = Vector{Float64}(undef, 0)
		for i in 1:N 
			push!(L, N * norm(A[i,:])^2/4)
		end
		γ = 10/(maximum(L))
		@testset "SVRG-Base" begin
			solver = StoInc.SVRG{T}(maxit=maxit, tol=1e-6, γ = γ,
			 verbose=true, freq=100000, report_data = (true, 10))
			@time x_SVRG, it_SVRG, ~, sol_hist = solver(F, g, x0, N=N)
			@test norm(x_SVRG-x_star) <= 1e-4 	
			@test it_SVRG == 375
		end 
		@testset "SVRG++" begin 
			solver = StoInc.SVRG{T}(maxit=maxit, tol=1e-6, γ = γ, m = N/4 |> Int,
			 verbose=true, freq=100000, report_data = (true, 1), plus= true)
			@time x_SVRG, it_SVRG, ~, sol_hist = solver(F, g, x0, N=N)
			@test norm(x_SVRG-x_star) <= 1e-4 	
			@test it_SVRG == 11
		end 
end
struct SVRG_basic_iterable{R <: Real, Tx, Tf, Tg}
	f:: Array{Tf}       # smooth term (for now  f_i =f  for all i) 
	g:: Tg         	 	# smooth term (for now  f_i =f  for all i) 
	x0::Tx            	# initial point
	N :: Int64        	# number of data points in the finite sum problem 
	L::Maybe{Union{Array{R},R}}   	# Lipschitz moduli of the gradients
	μ::Maybe{Union{Array{R},R}}   	# convexity moduli of the gradients
	γ::Maybe{R} 		# stepsizes 
	m::Maybe{Int64}		# number of inner loop updates
	plus::Bool 			# for SVRG++ variant 
end

mutable struct SVRG_basic_state{R <: Real, Tx}  # variables of the iteration, memory place holders for inplace operation, etc
	γ::R  					# average γ 
	m::Int64 				# number of inner loop updates
	av:: Tx 			 		# the running average
	z::Tx 					# for keeping track of the sum of w's
	z_full::Tx         	 	# the outer loop argument
	w::Tx         	 		# the inner loop variable
	ind::Array{Int64} 		# running ind set from which the algorithm chooses a coordinate 
	# some extra placeholders 
	∇f_temp::Tx 		 	# temporary placeholder for gradients 
	temp::Tx         	 	# for temporary computations
	rep1::Int64 			# only for now to report the progress! number of gradient updates
	rep2 					# only for now to report the progress!
	rep3::Int64 			# only for now to report the progress! counter
	rep4				# number of grads at which we saved
end

function SVRG_basic_state(γ::R, m::Int64, av::Tx, z::Tx, z_full::Tx, w::Tx, ind) where {R, Tx}
	return SVRG_basic_state{R, Tx}(γ, m, av, z, z_full, w, ind, copy(av), copy(av), 0.0, [], 0.0, [])
end

function Base.iterate(iter::SVRG_basic_iterable{R,Tx}) where {R, Tx}  # TODO: separate the case for the linesearch
	N   = iter.N
	ind = collect(1:N) 
	m = iter.m  === nothing ? m = N : m = iter.m # deviding here because we later multiply, for reporting without bug!
	# updating the stepsize 
	if iter.γ === nothing 
		if iter.L === nothing || iter.μ === nothing 
			@warn "--> smoothness or convexity parameter absent"
			return nothing
		else  
			L_M= maximum(iter.L)
			μ_M= maximum(iter.μ)

			γ =iter.α *10 / L_M
			println(γ)
			# condition Theorem 3.1
			rho = (1+ 4* L_M* γ^2*μ_M*(N+1) ) / (μ_M*γ*N*(1-4L_M*γ))
			@assert rho < 1
		end
	else 
		γ = iter.γ #provided γ
	end 
	#initializing the vectors 
	av = zero(iter.x0)
	for i in 1:N 	  # for loop to compute the individual nabla f_i for initialization
		∇f, ~ = gradient(iter.f[i], iter.x0) 
		∇f ./= N  	
		av .+= ∇f
	end
	z_full = copy(iter.x0)
	z = zero(av)  # this is placeholder for sum_{i=1}^m w_i
	w = copy(iter.x0)
	# println(m)
	state = SVRG_basic_state(γ, m, av, z, z_full, w, ind)
	return state, state
end

function Base.iterate(iter::SVRG_basic_iterable{R,Tx}, state::SVRG_basic_state{R, Tx}) where {R, Tx}
	# The inner cycle
	state.rep1 = 0
	for i in rand(state.ind, state.m)
		gradient!(state.temp, iter.f[i], state.z_full) # update the gradient
		gradient!(state.∇f_temp, iter.f[i], state.w) # update the gradient
		state.temp .-=  state.∇f_temp
		state.temp .-=  state.av
		state.temp .*= state.γ
		state.temp .+= state.w 
		prox!(state.w, iter.g, state.temp, state.γ)
		state.z .+= state.w   # keeping track of the sum of w's
		state.rep1 += 1 # only one gradient since we are reporting for Lasso or logistic
		state.rep3 += 1
		if mod(state.rep1, 1000) == 0
			push!(state.rep2, state.z ./ state.rep1)   
			push!(state.rep4, state.rep3)
		end
	end
	# full update 	
	state.z_full .=  state.z ./ state.m
	if !iter.plus state.w .=  state.z_full end # only for basic SVRG
	state.z  .= zero(state.z)  # for next iterate 
	state.av .= state.z
	for i in 1:iter.N 
		gradient!(state.∇f_temp, iter.f[i], state.z_full) # update the gradient
		state.∇f_temp ./= iter.N
		state.av .+= state.∇f_temp
		state.rep3 += 1 # --------remove 
	end
	if iter.plus state.m *= 2  end # only for plus SVRG
	return state, state
end 

#TODO list
## fundamental
#TODO list
## fundamental
	#### mini batch version of low memory
	#### res can be removed, for now I'm using it for testing only 
	#### fix reporting: should it be by the number of iterations? 

struct FINITO_LFinito_iterable{R <: Real, Tx, Tf, Tg}
	f:: Array{Tf}       # smooth term (for now  f_i =f  for all i) 
	g:: Tg         	 	# smooth term (for now  f_i =f  for all i) 
	x0::Tx            	# initial point
	N :: Int64        	# number of data points in the finite sum problem 
	L::Maybe{Array{R}}  # Lipschitz moduli of the gradients
	γ::Maybe{Union{Array{R},R}}  # stepsizes 
	α::R          		# in (0, 1), e.g.: 0.95
	sweeping::Int 		# to only use one stepsize γ
	single_stepsize::Bool 	# to only use one stepsize γ
end

mutable struct FINITO_LFinito_state{R <: Real, Tx}  # variables of the iteration, memory place holders for inplace operation, etc
	γ::Array{R}         		# stepsize parameter: can also go to iterable since it is constant throughout 
	hat_γ::R  					# average γ 
	av:: Tx 			 		# the running average
	ind::Array{Int64} 		# running ind set from which the algorithm chooses a coordinate 
	# some extra placeholders 
	z::Tx 						# zbar in the notes  
	∇f_temp::Tx 		 		# temporary placeholder for gradients 
	res::Tx         	 		# residual (to be decided)   # can be removed provided that I find a nicer way to terminate
	z_full::Tx         	 		# residual (to be decided)
end

function FINITO_LFinito_state(γ::Array{R}, hat_γ::R, av::Tx, ind) where {R, Tx}
	return FINITO_LFinito_state{R, Tx}(γ, hat_γ, av, ind,  copy(av), copy(av), copy(av).+R(1), copy(av))
end

function Base.iterate(iter::FINITO_LFinito_iterable{R,Tx}) where {R, Tx}  # TODO: separate the case for the linesearch
	N   = iter.N
	ind = collect(1:N) 
	# updating the stepsize 
	if iter.γ === nothing 
		if iter.L === nothing 
				@warn "--> smoothness parameter absent"
				return nothing
		else
			γ = zeros(N)  
			for i in 1:N
				iter.single_stepsize ? 
				(γ[i] =iter.α * iter.N / maximum(iter.L)) : (γ[i] =iter.α * iter.N /(iter.L[i]))
			end
		end
	else 
		isa(iter.γ,R) ? (γ = fill(iter.γ,(N,)) ) : (γ = iter.γ) #provided γ
	end 
	#initializing the vectors 
	hat_γ = 1/sum(1 ./ γ)
	av = copy(iter.x0)
	for i in 1:N 	  # for loop to compute the individual nabla f_i for initialization
		∇f, ~ = gradient(iter.f[i], iter.x0) 
		∇f .*=	hat_γ / N  	
		av .-= ∇f
	end
	state = FINITO_LFinito_state(γ, hat_γ, av, ind)
	return state, state
end

function Base.iterate(iter::FINITO_LFinito_iterable{R,Tx}, state::FINITO_LFinito_state{R, Tx}) where {R, Tx}
	# full update 
	prox!(state.z_full, iter.g, state.av, state.hat_γ)
	state.av .=  state.z_full
	for i in 1:iter.N 
		gradient!(state.∇f_temp, iter.f[i], state.z_full) # update the gradient
		state.av .-= state.hat_γ/iter.N .* state.∇f_temp
	end
	# The inner cycle
	if iter.sweeping == 2  # cyclic 
		state.ind =  1:iter.N
	elseif 	  iter.sweeping == 3 # shuffled cyclic 
		state.ind =  randperm(iter.N)
	end
	# println(state.ind)
	for i in state.ind 
		prox!(state.z, iter.g, state.av, state.hat_γ)
		gradient!(state.∇f_temp, iter.f[i], state.z_full) # update the gradient
		state.av .+=  (state.hat_γ / iter.N) .* state.∇f_temp
		gradient!(state.∇f_temp, iter.f[i], state.z) # update the gradient
		state.av .-=  (state.hat_γ / iter.N) .* state.∇f_temp
		state.av .+=   (state.hat_γ / state.γ[i]) .* (state.z .- state.z_full) 
	end
	return state, state
end 
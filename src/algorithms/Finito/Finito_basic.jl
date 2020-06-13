struct FINITO_basic_iterable{R <: Real, Tx, Tf, Tg}
	f:: Array{Tf}       # smooth term (for now  f_i =f  for all i) 
	g:: Tg         	 	# smooth term (for now  f_i =f  for all i) 
	x0::Tx            	# initial point
	N :: Int64        	# number of data points in the finite sum problem 
	L::Maybe{Array{R}}  # Lipschitz moduli of the gradients
	γ::Maybe{Union{Array{R},R}}  # stepsizes 
	α::R          		# in (0, 1), e.g.: 0.95
	sweeping::Int 		# update strategy: 1 for rand, 2 for cyclic, 3 for shuffled cyclic
	single_stepsize::Bool 	# to only use one stepsize γ
end

mutable struct FINITO_basic_state{R <: Real, Tx}  # variables of the iteration, memory place holders for inplace operation, etc
	p::Array{Tx}              	# table of x_j- γ_j/N nabla f_j(x_j) stacked as array of arrays	
	γ::Array{R}         		# stepsize parameter: can also go to iterable since it is constant throughout 
	hat_γ::R  					# average γ 
	av::Tx 			 			# the running average
	z::Tx 						# z as in manuscript
	ind::Array{Int64,1} 		# running ind set from which the algorithm chooses a coordinate 
	# some extra placeholders 
	∇f_temp::Tx 		 		# temporary placeholder for gradients 
	idxr::Int64 				# running idx in the iterate 
	idx::Int64 					# location of idxr in 1:N 
end

function FINITO_basic_state(p::Array{Tx}, γ::Array{R}, hat_γ::R, av::Tx, z::Tx, ind) where {R, Tx}
	return FINITO_basic_state{R, Tx}(p, γ, hat_γ, av, z, ind, zero(av), Int(1), Int(0))
end

function Base.iterate(iter::FINITO_basic_iterable{R,Tx}) where {R, Tx} 
	N =iter.N
	ind = collect(1:N) # full index set 
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
	# computing the gradients and updating the table p 
	p = Vector{Vector{Float64}}(undef, 0) # empty array 
	for i in 1:N 	  # for loop to compute the individual nabla f_i for initialization
	∇f, ~ = gradient(iter.f[i], iter.x0) 
	push!(p,iter.x0 - γ[i]/N*∇f) # table of x_i
	end
	#initializing the vectors 
	hat_γ = 1/sum(1 ./ γ)
	av = hat_γ*(sum(p ./ γ)) # the running average  
	z, ~ = prox(iter.g, av, hat_γ) 

	state = FINITO_basic_state(p, γ, hat_γ, av, z, ind)
	return state, state
end

function Base.iterate(iter::FINITO_basic_iterable{R,Tx}, state::FINITO_basic_state{R, Tx}) where {R, Tx}
	# manipulating indices 
	if iter.sweeping == 1 # uniformly random 	
		state.idxr = rand(1:iter.N) 
	elseif iter.sweeping == 2  # cyclic
		state.idxr = mod(state.idxr, iter.N)+1
	elseif  iter.sweeping == 3  # shuffled cyclic
		if  state.idx == iter.N
			state.ind = randperm(iter.N) 
			state.idx = 1
		else 
			state.idx += 1
		end
		state.idxr = state.ind[state.idx] 
	end 
	# perform the main steps 
	gradient!(state.∇f_temp, iter.f[state.idxr], state.z) # update the gradient
	state.∇f_temp .*= (state.γ[state.idxr]/iter.N) 
	state.z .-= state.∇f_temp # playing the rule of v here to become z again within prox
	@. state.av += (state.z - state.p[state.idxr]) * (state.hat_γ/state.γ[state.idxr])
	state.p[state.idxr] .= state.z  #update x_i
	prox!(state.z, iter.g, state.av, state.hat_γ)	

	return state, state
end 

#TODO list
## fundamental

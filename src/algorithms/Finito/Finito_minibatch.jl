struct FINITO_batch_iterable{R <: Real, Tx, Tf, Tg}
	f:: Array{Tf}       # smooth term (for now  f_i =f  for all i) 
	g:: Tg         	 	# smooth term (for now  f_i =f  for all i) 
	x0::Tx            	# initial point
	N :: Int64        	# number of data points in the finite sum problem 
	L::Maybe{Array{R}}  # Lipschitz moduli of the gradients
	γ::Maybe{Union{Array{R},R}}  # stepsizes 
	α::R          		# in (0, 1), e.g.: 0.95
	sweeping::Int 		# update strategy: 1 for rand, 2 for cyclic, 3 for shuffled cyclic
	single_stepsize::Bool 	# to only use one stepsize γ
	batch::Int64 		# batch size
end

mutable struct FINITO_batch_state{R <: Real, Tx}  # variables of the iteration, memory place holders for inplace operation, etc
	p::Array{Tx}              	# table of x_j- γ_j/N nabla f_j(x_j) stacked as array of arrays	
	γ::Array{R}         		# stepsize parameter: can also go to iterable since it is constant throughout 
	hat_γ::R  					# average γ 
	av:: Tx 			 		# the running average
	z::Tx 						# z as in manuscript
	ind::Array{Array{Int64}} 	# running ind set from which the algorithm chooses a coordinate 
	d::Int64 					# number of batches 
	# some extra placeholders 
	∇f_temp::Tx 		 		# temporary placeholder for gradients 
	idxr::Int64 				# running idx in the iterate 
	idx::Int64 					# location of idxr in 1:N 
	inds::Array{Int64}			# needed for shuffled only  
end

function FINITO_batch_state(p::Array{Tx}, γ::Array{R}, hat_γ::R, av, z, ind, d) where {R, Tx}
	return FINITO_batch_state{R, Tx}(p, γ, hat_γ, av, z, ind, d, zero(av), Int(1), Int(0), collect(1:d))
end

function Base.iterate(iter::FINITO_batch_iterable{R,Tx}) where {R, Tx}  # TODO: separate the case for the linesearch
	N =iter.N
	# define the batches
	r = iter.batch # batch size 
	# create index sets 
	if iter.sweeping == 1 
		ind = [collect(1:r)] # placeholder
	else 
		ind = Vector{Vector{Int64}}(undef, 0)
		d =  Int64(floor(N/r))
		for i in 1:d
			push!(ind, collect(r*(i-1)+1:i*r))
		end 
		if r*d < N push!(ind, collect(r*d+1:N)) end 
	end 
	d = cld(N,r) # number of batches  
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

	state = FINITO_batch_state(p, γ, hat_γ, av, z, ind, d)
	return state, state
end

function Base.iterate(iter::FINITO_batch_iterable{R,Tx}, state::FINITO_batch_state{R, Tx}) where {R, Tx}
	# manipulating indices 
	if iter.sweeping == 1 # uniformly random 	
		# state.idxr = rand(1:state.d) 
		state.idxr = 1
		state.ind = [rand(1:iter.N, iter.batch)]  
	elseif iter.sweeping == 2  # cyclic
		state.idxr = mod(state.idxr, state.d)+1
	elseif  iter.sweeping == 3  # shuffled cyclic
		if  state.idx == state.d
			state.inds = randperm(state.d) 
			state.idx = 1
		else 
			state.idx += 1
		end
		state.idxr = state.inds[state.idx] 
	end 
	# the iterate
	for i in state.ind[state.idxr]
		# perform the main steps 
		gradient!(state.∇f_temp, iter.f[i], state.z) # update the gradient
		state.∇f_temp .*= -(state.γ[i]/iter.N) 
		state.∇f_temp .+= state.z # playing the rule of v here to become z again within prox
		@. state.av += (state.∇f_temp - state.p[i]) * (state.hat_γ/state.γ[i])
		state.p[i] .= state.∇f_temp  #update x_i
	end
	prox!(state.z, iter.g, state.av, state.hat_γ)	

	return state, state
end 


#TODO list
## fundamental
	# renew the index set ind once a cycle is finished
	# the batches are static for now
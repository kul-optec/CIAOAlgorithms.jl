struct FINITO_ext_iterable{R <: Real, Tx, Tf, Tg}
	f:: Array{Tf}       # smooth term (for now  f_i =f  for all i) 
	g:: Tg         	 	# smooth term (for now  f_i =f  for all i) 
	x0::Tx            	# initial point
	N :: Int64        	# number of data points in the finite sum problem 
	L::Maybe{Array{R}}  # Lipschitz moduli of the gradients
	γ::Maybe{Union{Array{R},R}}  # stepsizes 
	α::R 	         	# in (0, 1), e.g.: 0.95
	tol::R 				# coordinate-wise tolerance
	sweeping::Int 		# update strategy: 1 for rand, 2 for cyclic, 3 for shuffled cyclic
	single_stepsize::Bool 	# to only use one stepsize γ
	μ::Maybe{Array{R}} # strong convexity moduli of the gradients for the str convex case
end

mutable struct FINITO_ext_state{R <: Real, Tx}  # variables of the iteration, memory place holders for inplace operation, etc
	p::Array{Tx}              	# table of x_j stacked as array of arrays	
	∇f::Array{Tx} 				# table of gradients 
	γ::Array{R}         		# stepsize parameter: can also go to iterable since it is constant throughout 
	hat_γ::R  					# average γ 
	indr::Array{Int64,1} 		# ind set from which the algorithm chooses a coordinate 
	av:: Tx 			 		# the running average
	z::Tx 						# zbar in the notes
	pi::Maybe{Array{R}}			# probabilities for the weigthed case   
	# some extra placeholders 
	res::Tx         	 		# residual (to be decided)
	ind::Array{Int64,1} 		# list of remaining coordinates (used for termination)
	idx::Int64 		 			# idx to be updated  
	idxr::Int64 				# running idx in the iterate 
end

function FINITO_ext_state(p::Array{Tx}, ∇f::Array{Tx}, γ::Array{R}, hat_γ::R, indr, av, z, pi) where {R, Tx}
	return FINITO_ext_state{R, Tx}(p, ∇f, γ, hat_γ, indr, av, z, pi, zero(av), copy(indr), Int(0), Int(1))
end

function Base.iterate(iter::FINITO_ext_iterable{R,Tx}) where {R, Tx}  # TODO: separate the case for the linesearch
	N =iter.N
	ind = collect(1:N) # full index set 

	p = Vector{Vector{Float64}}(undef, 0) # empty appropriate array with clean refrencing 	
	∇f = fill(iter.x0,(N,));    		
	for i in 1:N 	  # for loop to compute the individual nabla f_i for initialization
	∇f[i], ~ = gradient(iter.f[i], iter.x0) 
	push!(p,copy(iter.x0)) # table of x_i
	end
	# updating the stepsize 
	if iter.γ === nothing 
		if iter.L === nothing 
			@warn "--> smoothness parameter absent"
			return nothing
		else	
			γ = zeros(N)  
			pi = zeros(N)
			if iter.sweeping ==4 κ = iter.L ./ iter.μ end
			for i in 1:N
				if iter.sweeping == 4
					pi[i] = (sqrt(κ[i]) + sqrt(κ[i]-1) )^2
					γ[i] = (1 - sqrt(1-1/κ[i]) ) * iter.N / iter.μ[i] 
				else
					iter.single_stepsize ? 
					(γ[i] =iter.α * iter.N / maximum(iter.L)) : (γ[i] =iter.α * iter.N /(iter.L[i]))
				end
			end
		end
	else 
		isa(iter.γ,R) ? (γ = fill(iter.γ,(N,)) ) : (γ = iter.γ) #provided γ
	end
	#initializing the vectors 
	hat_γ = 1/sum(1 ./ γ)
	av = hat_γ*(sum(p ./ γ) - sum(∇f)/(length(ind)) ) # the running average  
	z, ~ = prox(iter.g, av, hat_γ) #### this is redundant when the line search is successful

	state = FINITO_ext_state(p, ∇f, γ, hat_γ, ind, av, z, pi)
	return state, state
end

function Base.iterate(iter::FINITO_ext_iterable{R,Tx}, state::FINITO_ext_state{R, Tx}) where {R, Tx}
	# manipulating indices 
	#if |z-x_i| is small no step is taken on that coordinate
	while true 
		# decide on the index 
		if iter.sweeping == 1 # uniformly random 	
			state.idxr = rand(state.ind) 
		elseif iter.sweeping == 2  # cyclic
			state.idxr = mod(state.idxr, iter.N)+1
		elseif  iter.sweeping == 3  # shuffled cyclic
			if  state.idx == iter.N
				if (length(state.ind) == iter.N) state.indr = randperm(iter.N) end 
				state.idx = 1
			else 
				state.idx += 1
			end
			state.idxr = state.indr[state.idx]
		elseif iter.sweeping ==4 #weighted probabilities: performs bad 
			temp2 = rand()*sum(state.pi[state.ind])  
			cnt = 0;
			while temp2 > 0
				cnt +=1
				temp2 -= state.pi[state.ind][cnt]
			end
			state.idxr = state.ind[cnt]
		end
		@. state.res = state.z - state.p[state.idxr] 
		if norm(state.res) >= (iter.tol * sqrt(state.γ[state.idxr])) break end 
		deleteat!(state.ind, findfst(state.ind,state.idxr))
		if state.ind == [] 
			return state, state   
		end 
	end
	state.ind = collect(1:iter.N)

	# perform the main steps 
	@. state.av += (state.hat_γ/ state.γ[state.idxr]) *(state.z - state.p[state.idxr])  
	state.p[state.idxr] .= state.z  #update x_i
	@. state.av += (state.hat_γ/iter.N)*state.∇f[state.idxr]
	gradient!(state.∇f[state.idxr], iter.f[state.idxr], state.z) # update the gradient
	@. state.av -= (state.hat_γ / iter.N)*state.∇f[state.idxr] 
	prox!(state.z, iter.g, state.av, state.hat_γ)	

	return state, state
end 

#TODO list
## fundamentals

## superficial

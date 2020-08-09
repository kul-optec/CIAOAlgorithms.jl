struct FINITO_adaptive_iterable{R<:Real,Tx,Tf,Tg}
    f::Array{Tf}			# smooth term  
    g::Tg          			# nonsmooth term 
    x0::Tx            		# initial point
    N::Int64        		# number of data points in the finite sum problem 
    L::Maybe{Union{Array{R},R}}  # Lipschitz moduli of nabla f_i    
    α::R          			# in (0, 1), e.g.: 0.99
    tol::R 					# coordinate-wise tolerance
    tol_ada::R 				# tolerance for the adaptive part 
    sweeping::Int 			# update strategy: 1 for rand, 2 for cyclic, 3 for shuffled cyclic
end

mutable struct FINITO_adaptive_state{R<:Real,Tx}  
    p::Array{Tx}			# table of x_j stacked as array of arrays	
    ∇f::Array{Tx} 			# table of gradients 
    γ::Array{R}         	# stepsize parameter 
    hat_γ::R  				# average γ 
    indr::Array{Int64,1} 	# ind set from which the algorithm chooses a coordinate 
    fi_x::Array{R,1}		# value of smooth term
    av::Tx  				# the running average
    z::Tx 					# zbar   
    # some extra placeholders 
    τ::R 					# stepsize for linesearch 
    res::Tx 				# residual (to be decided)
    γ_b::R  				#  
    tot_bt::R 				# placeholder for number of gradients used to estimate L_i 
    ind::Array{Int64,1} 	# list of remaining coordinates (used for termination)
    idx::Int64  			# idx to be updated  
    idxr::Int64 			# running idx
end

function FINITO_adaptive_state(
    p::Array{Tx},
    ∇f::Array{Tx},
    γ::Array{R},
    hat_γ::R,
    indr,
    fi_x,
    av,
    z,
) where {R,Tx}
    return FINITO_adaptive_state{R,Tx}(
        p,
        ∇f,
        γ,
        hat_γ,
        indr,
        fi_x,
        av,
        z,
        1.0,
        zero(av),
        0.0,
        0.0,
        copy(indr),
        Int(0),
        Int(0),
    )
end

function Base.iterate(iter::FINITO_adaptive_iterable{R,Tx}) where {R,Tx}  
    N = iter.N
    ind = collect(1:N) # full index set 
    # computing the gradients and updating the table p 
    p = Vector{Vector{Float64}}(undef, 0) 
    ∇f = fill(iter.x0, (N,))
    fi_x = rand(N)    	# compute the cost for the case of lineasearch 
    for i = 1:N			
        ∇f[i], fi_x[i] = gradient(iter.f[i], iter.x0)  
        push!(p, copy(iter.x0)) # table of x_i
    end
    # updating the stepsize 
    γ = zeros(N)
    for i = 1:N
        L_int = zeros(N)
        xeps = iter.x0 .+ one(R)   
        grad_f_xeps, f_xeps = gradient(iter.f[i], xeps)
        nmg = norm(grad_f_xeps - ∇f[i])
        t = 1
        while nmg < eps(R)  # in case xeps has the same gradient
            println("initial upper bound for L too small")
            xeps = iter.x0 .+ rand(t * [-1, 1], size(iter.x0))
            grad_f_xeps, f_xeps = gradient(iter.f[i], xeps)
            nmg = norm(grad_f_xeps - ∇f[i])
            t *= 2
        end
        # decide how to initialize! 
        L_int[i] = nmg / (t * sqrt(length(iter.x0)))
        L_int[i] /= iter.N
        γ[i] = iter.α / (L_int[i])
    end
    #initializing the vectors 
    hat_γ = 1 / sum(1 ./ γ)
    av = hat_γ * (sum(p ./ γ) - sum(∇f) / (length(ind))) # the running average  
    z, ~ = prox(iter.g, av, hat_γ)

    state = FINITO_adaptive_state(p, ∇f, γ, hat_γ, ind, fi_x, av, z)
    return state, state
end

function Base.iterate(
    iter::FINITO_adaptive_iterable{R,Tx},
    state::FINITO_adaptive_state{R,Tx},
) where {R,Tx}
    # index update and backtracking on smoothness moduli
    while true  # if |z-x_i| is small no step is taken on that coordinate
        # select an index 
        if iter.sweeping == 1 # uniformly random 	
            state.idxr = rand(state.ind)
        elseif iter.sweeping == 2  # cyclic
            state.idxr = mod(state.idxr, iter.N) + 1
        elseif iter.sweeping == 3  # shuffled cyclic
            if state.idx == iter.N
                if (length(state.ind) == iter.N)
                    state.indr = randperm(iter.N)
                end
                state.idx = 1
            else
                state.idx += 1
            end
            state.idxr = state.indr[state.idx]
        end

        @. state.res = state.z - state.p[state.idxr]
        # backtrack γ (warn if γ gets too small)   
        while true
            if state.γ[state.idxr] < iter.tol_ada / iter.N
                @warn "parameter `γ` became too small ($(state.γ))"
                return nothing
            end
            ~, fi_z = gradient(iter.f[state.idxr], state.z)
            state.tot_bt += 1
            fi_model =
                state.fi_x[state.idxr] +
                real(dot(state.∇f[state.idxr], state.res)) +
                (0.5 * iter.N * iter.α / state.γ[state.idxr]) * (norm(state.res)^2)
            tol = 10 * eps(R) * (1 + abs(fi_z))
            if fi_z <= fi_model + tol
                break
            end
            state.γ_b = state.γ[state.idxr]
            state.γ[state.idxr] *= 0.8
            # update hat_γ, av, z  
            state.av ./= state.hat_γ
            @. state.av += state.p[state.idxr] / state.γ[state.idxr]
            @. state.av -= state.p[state.idxr] / state.γ_b
            state.hat_γ = 1 / (1 / state.hat_γ + 1 / state.γ[state.idxr] - 1 / state.γ_b) 
            state.av .*= state.hat_γ
            prox!(state.z, iter.g, state.av, state.hat_γ) # compute prox(barz)    
            @. state.res = state.z - state.p[state.idxr]
        end
        if norm(state.res) >= (iter.tol .* sqrt(state.γ[state.idxr]))
            break
        end
        deleteat!(state.ind, findfst(state.ind, state.idxr))
        if state.ind == []
            return state, state
        end
    end
    state.ind = collect(1:iter.N)

    # perform the main steps 
    @. state.av += (state.hat_γ / state.γ[state.idxr]) * (state.z .- state.p[state.idxr])
    state.p[state.idxr] .= state.z  #update x_i
    @. state.av += (state.hat_γ / iter.N) * state.∇f[state.idxr]
    state.fi_x[state.idxr] = gradient!(state.∇f[state.idxr], iter.f[state.idxr], state.z) 
    @. state.av -= (state.hat_γ / iter.N) * state.∇f[state.idxr]
    prox!(state.z, iter.g, state.av, state.hat_γ)

    return state, state
end


#TODO list
## fundamental
#### ensuring envelope is lower bounded
#### the initial guess for L may be modified

struct FINITO_LFinito_iterable{R<:Real,Tx,Tf,Tg}
    f::Array{Tf}			# smooth term  
    g::Tg          			# nonsmooth term 
    x0::Tx            		# initial point
    N::Int64        		# number of data points in the finite sum problem 
    L::Maybe{Union{Array{R},R}}  # Lipschitz moduli of nabla f_i    
    γ::Maybe{Union{Array{R},R}}  # stepsizes 
    α::R          			# in (0, 1), e.g.: 0.99
    sweeping::Int 			# to only use one stepsize γ
end

mutable struct FINITO_LFinito_state{R<:Real,Tx}  
    γ::Array{R}         	# stepsize parameters
    hat_γ::R  				# average γ 
    av::Tx  				# the running average
    ind::Array{Int64} 		# running ind set for coordinate selection 
    # some extra placeholders 
    z::Tx 					  
    ∇f_temp::Tx  			# placeholder for gradients 
    z_full::Tx          
end

function FINITO_LFinito_state(γ::Array{R}, hat_γ::R, av::Tx, ind) where {R,Tx}
    return FINITO_LFinito_state{R,Tx}(
        γ,
        hat_γ,
        av,
        ind,
        copy(av),
        copy(av),
        copy(av),
    )
end

function Base.iterate(iter::FINITO_LFinito_iterable{R,Tx}) where {R,Tx} 
    N = iter.N
    ind = collect(1:N)
    # updating the stepsize 
    if iter.γ === nothing
        if iter.L === nothing
            @warn "--> smoothness parameter absent"
            return nothing
        else
            γ = zeros(N)
            for i = 1:N
                isa(iter.L, R) ? (γ = fill(iter.α * iter.N / iter.L, (N,))) :
                (γ[i] = iter.α * N / (iter.L[i]))
            end
        end
    else
        isa(iter.γ, R) ? (γ = fill(iter.γ, (N,))) : (γ = iter.γ) #provided γ
    end
    #initializing the vectors 
    hat_γ = 1 / sum(1 ./ γ)
    av = copy(iter.x0)
    for i = 1:N   
        ∇f, ~ = gradient(iter.f[i], iter.x0)
        ∇f .*= hat_γ / N
        av .-= ∇f
    end
    state = FINITO_LFinito_state(γ, hat_γ, av, ind)
    return state, state
end

function Base.iterate(
    iter::FINITO_LFinito_iterable{R,Tx},
    state::FINITO_LFinito_state{R,Tx},
) where {R,Tx}
    # full update 
    prox!(state.z_full, iter.g, state.av, state.hat_γ)
    state.av .= state.z_full
    for i = 1:iter.N
        gradient!(state.∇f_temp, iter.f[i], state.z_full) 
        state.av .-= state.hat_γ / iter.N .* state.∇f_temp
    end
    # The inner cycle
    if iter.sweeping == 2  # cyclic 
        state.ind = 1:iter.N
    elseif iter.sweeping == 3 # shuffled cyclic 
        state.ind = randperm(iter.N)
    end
    # println(state.ind)
    for i in state.ind
        prox!(state.z, iter.g, state.av, state.hat_γ)
        gradient!(state.∇f_temp, iter.f[i], state.z_full) 
        state.av .+= (state.hat_γ / iter.N) .* state.∇f_temp
        gradient!(state.∇f_temp, iter.f[i], state.z) 
        state.av .-= (state.hat_γ / iter.N) .* state.∇f_temp
        state.av .+= (state.hat_γ / state.γ[i]) .* (state.z .- state.z_full)
    end
    return state, state
end


#TODO list

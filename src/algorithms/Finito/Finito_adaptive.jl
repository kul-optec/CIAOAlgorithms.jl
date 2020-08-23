struct FINITO_adaptive_iterable{R<:Real,Tx,Tf,Tg}
    f::Array{Tf}            # smooth term  
    g::Tg                   # nonsmooth term 
    x0::Tx                  # initial point
    N::Int                  # of data points in the finite sum problem 
    L::Maybe{Union{Array{R},R}} # Lipschitz moduli of nabla f_i    
    tol::R                  # coordinate-wise tolerance
    tol_b::R                # tolerance for the adaptive part 
    sweeping::Int8          # 1, 2, 3 for rand, cyclic, shuffled
    α::R                    # in (0, 1), e.g.: 0.99
end

mutable struct FINITO_adaptive_state{R<:Real,Tx} <: AbstractFinitoState
    p::Array{Tx}            # table of x_j stacked as array of arrays	
    ∇f::Array{Tx}           # table of gradients 
    γ::Array{R}             # stepsize parameter 
    hat_γ::R                # average γ 
    indr::Array{Int}        # coordinate selection index set 
    fi_x::Array{R}          # value of smooth term
    av::Tx                  # the running average
    z::Tx                   # zbar   
    # some extra placeholders 
    τ::R                    # stepsize for linesearch 
    res::Tx                 # residual (for termination) 	
    γ_b::R
    ind::Array{Int}         # remaining coordinates (for termination)
    idx::Int                # idx to be updated  
    idxr::Int               # running idx
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
        R(1.0),
        zero(av),
        R(0.0),
        copy(indr),
        Int(0),
        Int(0),
    )
end

function Base.iterate(iter::FINITO_adaptive_iterable{R,Tx}) where {R,Tx}
    N = iter.N
    ind = collect(1:N) # full index set 
    # computing the gradients and updating the table p 
    p = Vector{Tx}(undef, 0)
    ∇f = fill(iter.x0, (N,))
    fi_x = zeros(R, N)    # compute the cost for the case of lineasearch 
    for i = 1:N
        ∇f[i], fi_x[i] = gradient(iter.f[i], iter.x0)
        push!(p, copy(iter.x0)) # table of x_i
    end
    # updating the stepsize 
    γ = zeros(R, N)
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

    # select an index 
    if iter.sweeping == 1 # uniformly random 	
        state.idxr = rand(1:iter.N)
    elseif iter.sweeping == 2  # cyclic
        state.idxr = mod(state.idxr, iter.N) + 1
    elseif iter.sweeping == 3  # shuffled cyclic
        if state.idx == iter.N
            state.ind = randperm(iter.N)
            state.idx = 1
        else
            state.idx += 1
        end
        state.idxr = state.ind[state.idx]
    end

    @. state.res = state.z - state.p[state.idxr]
    # backtrack γ (warn if γ gets too small)   
    while true
        if state.γ[state.idxr] < iter.tol_b / iter.N
            @warn "parameter `γ` became too small ($(state.γ))"
            return nothing
        end
        ~, fi_z = gradient(iter.f[state.idxr], state.z)
        fi_model =
            state.fi_x[state.idxr] +
            real(dot(state.∇f[state.idxr], state.res)) +
            (0.5 * iter.N * iter.α / state.γ[state.idxr]) * (norm(state.res)^2)
        tol = 10 * eps(R) * (1 + abs(fi_z))
        fi_z <= fi_model + tol && break

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
## the initial guess for L may be modified
## ensuring envelope is lower bounded
## minibatch not supported

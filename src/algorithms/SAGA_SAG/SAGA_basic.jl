struct SAGA_basic_iterable{R<:Real,C<:RealOrComplex{R},Tx<:AbstractArray{C},Tf,Tg}
    F::Array{Tf}            # smooth term  
    g::Tg                   # nonsmooth term 
    x0::Tx                  # initial point
    N::Int                  # of data points in the finite sum problem 
    L::Maybe{Union{Array{R},R}}  # Lipschitz moduli of nabla f_i	
    γ::Maybe{R}             # stepsize 
    SAG::Bool               # to activate SAG
end

mutable struct SAGA_basic_state{R<:Real,Tx}
    s::Array{Tx}            # table of x_j- γ_j/N nabla f_j(x_j) 
    γ::R                    # stepsize 
    av::Tx                  # the running average
    z::Tx
    # some extra placeholders 
    ind::Int         # running idx set 
    ∇f_temp::Tx             # placeholder for gradients 
    w::Tx
end

function SAGA_basic_state(s, γ::R, av::Tx, z::Tx) where {R,Tx}
    return SAGA_basic_state{R,Tx}(s, γ, av, z, 1, copy(av), copy(av))
end

function Base.iterate(iter::SAGA_basic_iterable{R,C,Tx}) where {R,C,Tx}
    N = iter.N
    # updating the stepsize 
    if iter.γ === nothing
        if iter.L === nothing
            @warn "smoothness parameter absent"
            return nothing
        else
            L_M = maximum(iter.L)
            γ = iter.SAG ? 1 / (16 * L_M) : γ = 1 / (3 * L_M)
        end
    else
        γ = iter.γ # provided γ
    end
    # computing the gradients and updating the table 
    s = Vector{Tx}(undef, 0)
    for i = 1:N
        ∇f, ~ = gradient(iter.F[i], iter.x0)
        push!(s, ∇f) # table of x_i
    end
    #initializing the vectors 
    av = sum(s) ./ N # the running average  
    z, ~ = prox(iter.g, (1 - γ) .* iter.x0, γ)
    state = SAGA_basic_state(s, γ, av, z)
    return state, state
end

function Base.iterate(iter::SAGA_basic_iterable{R}, state::SAGA_basic_state{R}) where {R}

    state.ind = rand(1:iter.N)
    gradient!(state.∇f_temp, iter.F[state.ind], state.z)
    if iter.SAG
        @. state.av += (state.∇f_temp - state.s[state.ind]) / iter.N
        @. state.w = state.z - state.γ * state.av
    else
        @. state.w = state.z - state.γ * (state.∇f_temp - state.s[state.ind] + state.av)
        @. state.av += (state.∇f_temp - state.s[state.ind]) / iter.N
    end
    prox!(state.z, iter.g, state.w, state.γ)
    state.s[state.ind] .= state.∇f_temp  #update x_i

    return state, state
end


solution(state::SAGA_basic_state) = state.z


#TODO: minibatch

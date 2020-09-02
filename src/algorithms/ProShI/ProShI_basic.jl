struct Proshi_basic_iterable{R<:Real,C<:Union{R,Complex{R}},Tx<:AbstractArray{C},Tf,Tg}
    f::Array{Tf}            # smooth term  
    g::Tg                   # nonsmooth term 
    x0::Tx                  # initial point
    N::Int                  # number of data points in the finite sum problem 
    L::Maybe{Union{Array{R},R}}  # Lipschitz moduli of nabla f_i	
    γ::Maybe{Union{Array{R},R}}  # stepsizes 
    sweeping::Int8          # to only use one stepsize γ
    batch::Int              # batch size
    α::R                    # in (0, 1), e.g.: 0.99
end

mutable struct Proshi_basic_state{R<:Real,Tx}
    s::Array{Tx}            # table of x_j- γ_j/N nabla f_j(x_j) 
    γ::Array{R}             # stepsize parameters 
    hat_γ::R                # average γ 
    av::Tx                  # the running average
    z::Tx
    ind::Array{Array{Int}} # running index set 
    # some extra placeholders 
    d::Int                  # number of batches 
    ∇f_temp::Tx             # placeholder for gradients 
    idxr::Int               # running idx in the iterate 
    idx::Int                # location of idxr in 1:N 
    inds::Array{Int}        # needed for shuffled only  
end

function Proshi_basic_state(s, γ, hat_γ::R, av::Tx, z::Tx, ind, d) where {R,Tx}
    return Proshi_basic_state{R,Tx}(
        s,
        γ,
        hat_γ,
        av,
        z,
        ind,
        d,
        zero(av),
        Int(1),
        Int(0),
        collect(1:d),
    )
end

function Base.iterate(iter::Proshi_basic_iterable{R,C,Tx}) where {R,C,Tx}
    N = iter.N
    # define the batches
    r = iter.batch # batch size 
    # create index sets 
    if iter.sweeping == 1
        ind = [collect(1:r)] # placeholder
    else
        ind = Vector{Vector{Int}}(undef, 0)
        d = Int(floor(N / r))
        for i = 1:d
            push!(ind, collect(r*(i-1)+1:i*r))
        end
        r * d < N && push!(ind, collect(r*d+1:N))
    end
    d = cld(N, r) # number of batches  
    # updating the stepsize 
    if iter.γ === nothing
        if iter.L === nothing
            @warn "--> smoothness parameter absent"
            return nothing
        else
            γ = zeros(R, N)
            for i = 1:N
                isa(iter.L, R) ? (γ = fill(iter.α * R(iter.N) / iter.L, (N,))) :
                (γ[i] = iter.α * R(N) / iter.L[i])
            end
        end
    else
        isa(iter.γ, R) ? (γ = fill(iter.γ, (N,))) : (γ = iter.γ) #provided γ
    end
    # computing the gradients and updating the table s 
    s = Vector{Tx}(undef, 0)
    for i = 1:N
        ∇f, ~ = gradient(iter.f[i], iter.x0)
        push!(s, iter.x0 - γ[i] / N * ∇f) # table of x_i
    end
    #initializing the vectors 
    hat_γ = sum(γ)
    av = sum(s) # the running average  
    z, ~ = prox(iter.g, av, hat_γ) # w in [1]
    z .-= av
    z ./= hat_γ
    state = Proshi_basic_state(s, γ, hat_γ, av, z, ind, d)

    return state, state
end

function Base.iterate(
    iter::Proshi_basic_iterable{R},
    state::Proshi_basic_state{R},
) where {R}
    # manipulating indices 
    if iter.sweeping == 1 # uniformly random    
        state.ind = [rand(1:iter.N, iter.batch)]
    elseif iter.sweeping == 2  # cyclic
        state.idxr = mod(state.idxr, state.d) + 1
    elseif iter.sweeping == 3  # shuffled cyclic
        if state.idx == state.d
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
        state.av .-= state.s[i]
        @. state.s[i] += state.γ[i] * state.z
        gradient!(state.∇f_temp, iter.f[i], state.s[i])
        state.∇f_temp .*= -(state.γ[i] / iter.N)
        state.∇f_temp .+= state.s[i]
        state.av .+= state.∇f_temp
        state.s[i] .= state.∇f_temp  #update x_i
    end
    prox!(state.z, iter.g, state.av, state.hat_γ)
    state.z .-= state.av
    state.z ./= state.hat_γ
    return state, state
end

function solution(state::Proshi_basic_state)
    for i = 1:length(state.s)
        state.s[i] .+= state.γ[i] * state.z
    end
    state.s
end

#TODO list
## in cyclic/shuffled minibatchs are static  

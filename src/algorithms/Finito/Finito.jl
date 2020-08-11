# Latafat, Themelis, Patrinos, "Block-coordinate and incremental aggregated
# proximal gradient methods for nonsmooth nonconvex problems."
# arXiv:1906.10053 (2019).
#
# Latafat. "Distributed proximal algorithms for large-scale structured optimization"
# PhD thesis, KU Leuven, 7 2020.
# 
# Mairal, "Incremental majorization-minimization optimization with application to
# large-scale machine learning."
# SIAM Journal on Optimization 25, 2 (2015), 829–855.
# 
# Defazio, Domke, "Finito: A faster, permutable incremental gradient method
# for big data problems."
# In International Conference on Machine Learning (2014), pp. 1125-1133.
#



using LinearAlgebra
using ProximalOperators
using ProximalAlgorithms.IterationTools
using Printf
using Base.Iterators
using Random


include("Finito_basic.jl")
include("Finito_LFinito.jl")
include("Finito_adaptive.jl")


struct Finito{R<:Real}
    γ::Maybe{Union{Array{R},R}}
    sweeping::Int
    LFinito::Bool
    extended::Bool
    adaptive::Bool
    minibatch::Tuple{Bool,Int}
    maxit::Int
    verbose::Bool
    freq::Int
    α::R
    tol::R
    tol_b::R
    function Finito{R}(;
        γ::Maybe{Union{Array{R},R}} = nothing,
        sweeping::Int = 1,
        LFinito::Bool = false,
        extended::Bool = false,
        adaptive::Bool = false,
        minibatch::Tuple{Bool,Int} = (false, 1),
        maxit::Int = 10000,
        verbose::Bool = false,
        freq::Int = 10000,
        α::R = R(0.999),
        tol::R = R(1e-8),
        tol_b::R = R(1e-9),
    ) where {R}
        @assert γ === nothing || minimum(γ) > 0
        @assert maxit > 0
        @assert tol > 0
        @assert tol_b > 0
        @assert freq > 0
        new(
            γ,
            sweeping,
            LFinito,
            extended,
            adaptive,
            minibatch,
            maxit,
            verbose,
            freq,
            α,
            tol,
            tol_b,
        )
    end
end

function (solver::Finito{R})(f, g, x0; L = nothing, N = N) where {R}

    stop(state) = false
    stop(state::FINITO_adaptive_state) = isempty(state.ind)

    disp(it, state) = @printf "%5d | %.3e  \n" it state.hat_γ

    # dispatching the structure
    if solver.LFinito
        iter = FINITO_LFinito_iterable(f, g, x0, N, L, solver.γ,
                solver.sweeping, solver.minibatch[2], solver.α,
        )
    elseif solver.adaptive
        iter = FINITO_adaptive_iterable(f, g, x0, N, L, solver.tol,
                solver.tol_b, solver.sweeping, solver.α,
        )
    else
        iter = FINITO_basic_iterable(f, g, x0, N, L, solver.γ,
                solver.sweeping, solver.minibatch[2], solver.α,
        )
    end

    iter = take(halt(iter, stop), solver.maxit)
    iter = enumerate(iter)

    num_iters, state_final = nothing, nothing
    for (it_, state_) in iter  # unrolling the iterator 
        # see https://docs.julialang.org/en/v1/manual/interfaces/index.html
        if solver.verbose && mod(it_, solver.freq) == 0
            disp(it_, state_)
        end
        num_iters, state_final = it_, state_
    end
    if solver.verbose && mod(num_iters, solver.freq) !== 0
        disp(num_iters, state_final)
    end # for the final iteration
    return state_final.z, num_iters
end

"""
    Finito([γ, sweeping, LFinito, adaptive, minibatch, maxit, verbose, freq, tol, tol_b])

Instantiate the Finito algorithm for solving fully nonconvex optimization problems of the form
    
    minimize 1/N sum_{i=1}^N f_i(x) + g(x)

    where `f_i` are smooth and `g` is possibly nonsmooth, all of which may be nonconvex.  

If `solver = Finito(args...)`, then the above problem is solved with

	solver(F, g, x0, N, L)

	where F is an array containing f_i's, x0 is the initial point, and L is an array of 
    smoothness moduli of f_i's; it is optional when γ is provided or in the adaptive case. 

Optional keyword arguments are:
* `γ`: an array of N stepsizes for each coordinate 
* `sweeping::Int` 1 for uniform randomized (default), 2 for cyclic, 3 for shuffled 
* `LFinito::Bool` low memory variant of the Finito/MISO algorithm
* `adaptive::Bool` to activate adaptive smoothness parameter computation
* `minibatch::(Bool,Int)` to use batchs of a given size    
* `maxit::Integer` (default: `10000`), maximum number of iterations to perform.
* `verbose::Bool` (default: `true`), whether or not to print information during the iterations.
* `freq::Integer` (default: `10000`), frequency of verbosity.
* `α::R` parameter where γ_i = αN/L_i
* `tol::Real` (default: `1e-8`), absolute tolerance for the adaptive case
* `tol_b::R` tolerance for the backtrack (default: `1e-9`)
"""

Finito(::Type{R}; kwargs...) where {R} = Finito{R}(; kwargs...)
Finito(; kwargs...) = Finito(Float64; kwargs...)

#TODO list

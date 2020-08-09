# Latafat, Themelis, Patrinos, "Block-coordinate and incremental aggregated
# proximal gradient methods for nonsmooth nonconvex problems."
# arXiv:1906.10053 (2019).
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
include("Finito_LFinito_minibatch.jl")
include("Finito_adaptive.jl")


struct Finito{R<:Real}
    γ::Maybe{Union{Array{R},R}}
    sweeping::Int
    LFinito::Bool
    extended::Bool
    adaptive::Bool
    minibatch::Tuple{Bool,Int}
    maxit::Int
    tol::R
    verbose::Bool
    freq::Int
    α::R
    tol_ada::R
    function Finito{R}(;
        γ::Maybe{Union{Array{R},R}} = nothing,
        sweeping::Int = 1,
        LFinito::Bool = false,
        extended::Bool = false,
        adaptive::Bool = false,
        minibatch::Tuple{Bool,Int} = (false, 1),
        maxit::Int = 10000,
        tol::R = R(1e-8),
        verbose::Bool = true,
        freq::Int = 10000,
        α::R = 0.999,
        tol_ada::R = 1e-9,
    ) where {R}
        @assert γ === nothing || minimum(γ) > 0
        @assert maxit > 0
        @assert tol > 0
        @assert freq > 0
        new(
            γ,
            sweeping,
            LFinito,
            extended,
            adaptive,
            minibatch,
            maxit,
            tol,
            verbose,
            freq,
            α,
            tol_ada,
        )
    end
end

function (solver::Finito{R})(f, g, x0; L = nothing, N = N) where {R}

    stop(state) = false 
    stop(state::FINITO_adaptive_state) = isempty(state.ind)

    disp(it, state) = @printf "%5d | %.3e  \n" it state.hat_γ

    # dispatching the structure
    if !solver.adaptive
        if solver.LFinito
            solver.minibatch[1] ?
            (
                iter = FINITO_LFinito_batch_iterable(
                    f,
                    g,
                    x0,
                    N,
                    L,
                    solver.γ,
                    solver.α,
                    solver.sweeping,
                    solver.minibatch[2],
                )
            ) :
            (
                iter = FINITO_LFinito_iterable(
                    f,
                    g,
                    x0,
                    N,
                    L,
                    solver.γ,
                    solver.α,
                    solver.sweeping,
                )
            )
        # elseif solver.minibatch[1]
        #     iter = FINITO_batch_iterable(
        #         f,
        #         g,
        #         x0,
        #         N,
        #         L,
        #         solver.γ,
        #         solver.α,
        #         solver.sweeping,
        #         solver.minibatch[2],
        #     )
        else
            iter = FINITO_basic_iterable(f, g, x0, N, L, solver.γ, solver.α, solver.sweeping, solver.minibatch[2])
        end
    else
        iter = FINITO_adaptive_iterable(
            f,
            g,
            x0,
            N,
            L,
            solver.α,
            solver.tol,
            solver.tol_ada,
            solver.sweeping,
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
    Finito([γ, sweeping, LFinito, adaptive, minibatch, maxit, tol, verbose, freq])

Instantiate the Finito algorithm for solving fully nonconvex optimization problems of the form
    
    minimize 1/N sum_{i=1}^N f_i(x) + g(x) 

If `solver = Finito(args...)`, then the above problem is solved with

	solver(F, g, x0, N, L)

	where L is optional depending on if stepsizes are provided to the solver, and F is an array containing f_i's.   

Optional keyword arguments are:
* `γ`: an array of N stepsizes for each coordinate 
* `L`: an array of smoothness moduli of f_i's 
* `sweeping::Bool` 1 for uniform randomized (default), 2 for cyclic, 3 for shuffled 
* `adaptive::Bool` to activate adaptive smoothness parameter computation
* `maxit::Integer` (default: `10000`), maximum number of iterations to perform.
* `tol::Real` (default: `1e-8`), absolute tolerance on the fixed-point residual.
* `verbose::Bool` (default: `true`), whether or not to print information during the iterations.
* `freq::Integer` (default: `100`), frequency of verbosity.
* `α::R` parameter where γ_i = αN/L_i
* `tol_ada::R` tolerance for the backtrack (default: `1e-9`)
"""

Finito(::Type{R}; kwargs...) where {R} = Finito{R}(; kwargs...)
Finito(; kwargs...) = Finito(Float64; kwargs...)

#TODO list

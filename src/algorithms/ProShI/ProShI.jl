# Latafat, Themelis, Patrinos, "Block-coordinate and incremental aggregated
# proximal gradient methods for nonsmooth nonconvex problems."
# arXiv:1906.10053 (2019).
#

using LinearAlgebra
using ProximalOperators
using ProximalAlgorithms.IterationTools
using Printf
using Base.Iterators
using Random

export solution

include("ProShI_basic.jl")

struct Proshi{R<:Real}
    γ::Maybe{Union{Array{R},R}}
    sweeping::Int8
    minibatch::Tuple{Bool,Int}
    maxit::Int
    verbose::Bool
    freq::Int
    α::R
    function Proshi{R}(;
        γ::Maybe{Union{Array{R},R}} = nothing,
        sweeping = 1,
        minibatch::Tuple{Bool,Int} = (false, 1),
        maxit::Int = 10000,
        verbose::Bool = false,
        freq::Int = 10000,
        α::R = R(0.999),
    ) where {R}
        @assert γ === nothing || minimum(γ) > 0
        @assert maxit > 0
        @assert freq > 0
        new(γ, sweeping, minibatch, maxit, verbose, freq, α)
    end
end

function (solver::Proshi{R})(
    x0::AbstractArray{C};
    F = nothing,
    g = ProximalOperators.Zero(),
    L = nothing,
    N = N,
) where {R,C<:RealOrComplex{R}}

    stop(state) = false

    disp(it, state) = @printf "%5d | %.3e  \n" it state.hat_γ

    F === nothing && (F = fill(ProximalOperators.Zero(), (N,)))
    # dispatching the iterator
    iter = Proshi_basic_iterable(
        F,
        g,
        x0,
        N,
        L,
        solver.γ,
        solver.sweeping,
        solver.minibatch[2],
        solver.α,
    )

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
    return solution(state_final), num_iters
end

"""
    Proshi([γ, sweeping, LProshi, adaptive, minibatch, maxit, verbose, freq, tol, tol_b])

Instantiate the Proshi algorithm for solving fully nonconvex optimization problems of the form
    
    minimize 1/N sum_{i=1}^N f_i(x) + g(x)

where `f_i` are smooth and `g` is possibly nonsmooth, all of which may be nonconvex.  

If `solver = Proshi(args...)`, then the above problem is solved with

	solver(x0, [F, g, N, L])

where F is an array containing f_i's, x0 is the initial point, and L is an array of 
smoothness moduli of f_i's; it is optional in the adaptive mode or if γ is provided. 

Optional keyword arguments are:
* `γ`: an array of N stepsizes for each coordinate 
* `sweeping::Int` 1 for uniform randomized (default), 2 for cyclic, 3 for shuffled 
* `LProshi::Bool` low memory variant of the Proshi/MISO algorithm
* `adaptive::Bool` to activate adaptive smoothness parameter computation
* `minibatch::(Bool,Int)` to use batchs of a given size    
* `maxit::Integer` (default: `10000`), maximum number of iterations to perform.
* `verbose::Bool` (default: `true`), whether or not to print information during the iterations.
* `freq::Integer` (default: `10000`), frequency of verbosity.
* `α::R` parameter where γ_i = αN/L_i
* `tol::Real` (default: `1e-8`), absolute tolerance for the adaptive case
* `tol_b::R` tolerance for the backtrack (default: `1e-9`)
"""

Proshi(::Type{R}; kwargs...) where {R} = Proshi{R}(; kwargs...)
Proshi(; kwargs...) = Proshi(Float64; kwargs...)


"""
If `solver = Proshi(args...)`, then 

    itr = iterator(solver, x0, [F, g, N, L])

is an iterable object. Note that [maxit, verbose, freq] fields of the solver are ignored here. 

The solution at any given state can be obtained using solution(state), e.g., 
for state in Iterators.take(itr, maxit)
    # do something using solution(state)
end

See https://docs.julialang.org/en/v1/manual/interfaces/index.html 
and https://docs.julialang.org/en/v1/base/iterators/ for a list of iteration utilities
"""


function iterator(
    solver::Proshi{R},
    x0::AbstractArray{C};
    F = nothing,
    g = ProximalOperators.Zero(),
    L = nothing,
    N = N,
) where {R,C<:RealOrComplex{R}}
    F === nothing && (F = fill(ProximalOperators.Zero(), (N,)))
    # dispatching the iterator
    iter = Proshi_basic_iterable(
        F,
        g,
        x0,
        N,
        L,
        solver.γ,
        solver.sweeping,
        solver.minibatch[2],
        solver.α,
    )
    return iter
end

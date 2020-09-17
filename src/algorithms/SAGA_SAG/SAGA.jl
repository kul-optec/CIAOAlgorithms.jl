# Defazio, Bach, Lacoste-Julien: SAGA: A fast incremental gradient method
# with support for non-strongly convex composite objectives. 
# In: Advances in neural information processing systems, pp. 1646–1654 (2014).
#
# Reddi, Sra, Poczos, and Smola, "Proximal stochastic methods for nonsmooth
# nonconvex finite-sum optimization." In Advances in Neural Information 
# Processing Systems, pp. 1145–1153 (2016).
#
# Schmidt, Le Roux, Bach, "Minimizing finite sums with the stochastic average gradient"
# Mathematical Programming, 162(1-2), 83-112 (2017).
# 

using LinearAlgebra
using ProximalOperators
using ProximalAlgorithms.IterationTools
using Printf
using Base.Iterators
using Random

export solution

include("SAGA_basic.jl")

struct SAGA{R<:Real}
    γ::Maybe{R}
    maxit::Int
    verbose::Bool
    freq::Int
    SAG_flag::Bool
    function SAGA{R}(;
        γ::Maybe{R} = nothing,
        maxit::Int = 10000,
        verbose::Bool = false,
        freq::Int = 1000,
        SAG_flag::Bool = false,
    ) where {R}
        @assert γ === nothing || γ > 0
        @assert maxit > 0
        @assert freq > 0
        new(γ, maxit, verbose, freq, SAG_flag)
    end
end

function (solver::SAGA{R})(
    x0::AbstractArray{C};
    F = nothing,
    g = ProximalOperators.Zero(),
    L = nothing,
    N = N,
) where {R,C<:RealOrComplex{R}}

    stop(state::SAGA_basic_state) = false
    disp(it, state) = @printf "%5d | %.3e  \n" it state.γ

    F === nothing && (F = fill(ProximalOperators.Zero(), (N,)))

    # dispatching the structure
    iter = SAGA_basic_iterable(F, g, x0, N, L, solver.γ, solver.SAG_flag)
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
    SAGA([γ, maxit, verbose, freq])

Instantiate the SAGA algorithm  for solving convex optimization problems of the form
    
    minimize 1/N sum_{i=1}^N f_i(x) + g(x) 

If `solver = SAGA(args...)`, then the above problem is solved with

	solver(x0, [F, g, N, L])

where F is an array containing f_i's, x0 is the initial point, and L is an array of 
smoothness moduli of f_i's; it is optional when γ is provided.  

Optional keyword arguments are:
* `γ`: stepsize  
* `L`: an array of smoothness moduli of f_i's 
* `maxit::Integer` (default: `10000`), maximum number of iterations to perform.
* `verbose::Bool` (default: `true`), whether or not to print information during the iterations.
* `freq::Integer` (default: `100`), frequency of verbosity.

References:

[1] Defazio, Bach, Lacoste-Julien, "SAGA: A fast incremental gradient method
with support for non-strongly convex composite objectives" 
In: Advances in neural information processing systems, pp. 1646–1654 (2014).

[2] Reddi, Sra, Poczos, and Smola, "Proximal stochastic methods for nonsmooth
nonconvex finite-sum optimization" In Advances in Neural Information 
Processing Systems, pp. 1145–1153 (2016).
"""

SAGA(::Type{R}; kwargs...) where {R} = SAGA{R}(; kwargs...)
SAGA(; kwargs...) = SAGA(Float64; kwargs...)


"""
If `solver = SAGA(args...)`, then 

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
    solver::SAGA{R},
    x0::AbstractArray{C};
    F = nothing,
    g = ProximalOperators.Zero(),
    L = nothing,
    N = N,
) where {R,C<:RealOrComplex{R}}

    F === nothing && (F = fill(ProximalOperators.Zero(), (N,)))
    # dispatching the iterator
    iter = SAGA_basic_iterable(F, g, x0, N, L, solver.γ, solver.SAG_flag)

    return iter
end



"""
    SAG([γ, maxit, verbose, freq])

Instantiate the SAG algorithm  for solving convex optimization problems of the form
    
    minimize 1/N sum_{i=1}^N f_i(x) + g(x) 

If `solver = SAG(args...)`, then the above problem is solved with

    solver(x0, [F, g, N, L])

where F is an array containing f_i's, x0 is the initial point, and L is an array of 
smoothness moduli of f_i's; it is optional when γ is provided.  

Optional keyword arguments are:
* `γ`: stepsize  
* `L`: an array of smoothness moduli of f_i's 
* `maxit::Integer` (default: `10000`), maximum number of iterations to perform.
* `verbose::Bool` (default: `true`), whether or not to print information during the iterations.
* `freq::Integer` (default: `100`), frequency of verbosity.

References:

[1] Schmidt, Le Roux, Bach, "Minimizing finite sums with the stochastic average gradient"
Mathematical Programming, 162(1-2), 83-112 (2017).

"""

"""
If `solver = SAG(args...)`, then 

    itr = iterator(solver, x0, [F, g, N, L])

is an iterable object. Note that [maxit, verbose, freq] fields of the solver are ignored here. 

The solution at any given state can be obtained using solution(state), e.g., 
for state in Iterators.take(itr, maxit)
    # do something using solution(state)
end

See https://docs.julialang.org/en/v1/manual/interfaces/index.html 
and https://docs.julialang.org/en/v1/base/iterators/ for a list of iteration utilities
"""

SAG(::Type{R}; kwargs...) where {R} = SAGA{R}(; kwargs..., SAG_flag = true)
SAG(; kwargs...) = SAG(Float64; kwargs...)

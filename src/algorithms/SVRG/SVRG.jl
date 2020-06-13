# Xiao, Zhang, "A proximal stochastic gradient method 
# with progressive variance reduction.",
# SIAM Journal on Optimization 24.4 (2014): 2057-2075.

using LinearAlgebra
using ProximalOperators
using ProximalAlgorithms.IterationTools
using Printf
using Base.Iterators
using Random

include("SVRG_basic.jl")

struct SVRG{R<:Real}
    γ::Maybe{R}   
    maxit::Int
    verbose::Bool
    freq::Int
    m::Maybe{Int} 
    plus::Bool
    function SVRG{R}(;γ::Maybe{R}= nothing, maxit::Int=10000, verbose::Bool=true, 
        freq::Int=1000, m::Maybe{Int} = nothing, plus::Bool=false
        ) where R
    	@assert γ === nothing || γ > 0
        @assert maxit > 0
        @assert freq > 0
        new(γ, maxit, verbose, freq, m, plus )
    end
end

function (solver::SVRG{R})( f, g, x0; L=nothing, μ=nothing, N = N) where {R}

	stop(state::SVRG_basic_state) = false 
 	disp(it, state) = @printf "%5d | %.3e  \n" it state.γ 

	m =  solver.m === nothing ? m = N : m = solver.m

	maxit = solver.plus ? floor(log2(cld(solver.maxit,m)+1)) |>Int : cld(solver.maxit,(N+m))   
	# dispatching the structure
	iter = SVRG_basic_iterable(f, g, x0, N, L, μ, solver.γ, m, solver.plus)
	iter = take(halt(iter, stop), maxit) 
	iter = enumerate(iter)
	num_iters, state_final = nothing, nothing
	for (it_, state_) in iter  	# unrolling the iterator 
		# see https://docs.julialang.org/en/v1/manual/interfaces/index.html
		if solver.verbose && mod(it_,solver.freq)==0 	disp(it_,state_)	end
   		num_iters, state_final = it_, state_
	end
    if solver.verbose && mod(num_iters,solver.freq) !== 0
 		disp(num_iters,state_final)
 	 end # for the final iteration
	return state_final.z_full, num_iters
end


"""
    SVRG([γ, maxit, verbose, freq, α, m, plus ])

Instantiate the SVRG algorithm  for solving (strongly) convex optimization problems of the form
    
    minimize 1/N sum_{i=1}^N f_i(x) + g(x) 

If `solver = SVRG(args...)`, then the above problem is solved with

	solver(F, g, x0, N, L, μ)

	where L and μ are optional depending on if stepsizes are provided or not, F is an array containing f_i's.   

Optional keyword arguments are:
* `γ`: an array of N stepsizes for each coordinate 
* `L`: an array of smoothness moduli of f_i's 
* `μ`: (if strongly convex) an array of strong convexity moduli of f_i's 
* `maxit::Integer` (default: `10000`), maximum number of iterations to perform.
* `verbose::Bool` (default: `true`), whether or not to print information during the iterations.
* `freq::Integer` (default: `100`), frequency of verbosity.
* `α::R` parameter for stepsizes!

"""


#TODO list
 	# decide how maxit should be adjust if at all
    # remove α

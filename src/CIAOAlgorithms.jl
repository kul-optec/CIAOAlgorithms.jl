module CIAOAlgorithms

const RealOrComplex{R} = Union{R, Complex{R}}

const ArrayOrTuple{R} = Union{
	AbstractArray{C, N} where {C <: RealOrComplex{R}, N},
	Tuple{Vararg{AbstractArray{C, N} where {C <: RealOrComplex{R}, N}}}
}

const Maybe{T} = Union{T, Nothing}

# utulities 
include("utilities/indexingUtilities.jl")

# algorithms 
include("algorithms/Finito/Finito.jl")
include("algorithms/SVRG/SVRG.jl")

end # module
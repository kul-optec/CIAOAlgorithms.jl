module CIAOAlgorithms

const Maybe{T} = Union{T, Nothing}

# utulities 
include("utilities/indexingUtilities.jl")

# algorithms 
include("algorithms/Finito/Finito.jl")
include("algorithms/SVRG/SVRG.jl")

end # module
